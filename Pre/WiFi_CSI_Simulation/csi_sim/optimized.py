"""Causal multiscale features, endpoint supervision and validation-only tuning."""
from __future__ import annotations

from pathlib import Path
import json

import h5py
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_cache, extract_windows
from .features import FEATURE_NAMES
from .fusion import TOPOLOGIES, fuse
from .utils import write_csv, write_json

MULTISCALE_STATS = ("variance_median", "variance_p90", "lagcov_median", "lagcov_p90",
                    "acf_mean", "active_tone_fraction", "group_variance_median", "group_variance_p90")


def multiscale_features(amplitude, noise, rate, windows):
    """Input is [decision, link, tone, past sample]; uses no labels or trajectory."""
    x = np.asarray(amplitude, dtype=np.float64)
    noise = np.maximum(np.asarray(noise, dtype=float), 1e-8)[None, :, None]
    values = []
    for seconds in windows:
        a = x[..., -round(seconds * rate):]
        centered = a - a.mean(axis=-1, keepdims=True)
        variance = np.mean(centered ** 2, axis=-1)
        covariance = np.mean(centered[..., :-1] * centered[..., 1:], axis=-1)
        for item in (variance, np.maximum(covariance, 0)):
            normalized = np.log1p(item / noise)
            values.extend((np.median(normalized, axis=-1), np.quantile(normalized, 0.9, axis=-1)))
        values.append(np.mean(covariance / np.maximum(variance, 1e-8), axis=-1))
        values.append(np.mean(variance > 2 * noise, axis=-1))
        groups = np.stack([np.var(a[:, :, ids].mean(axis=2), axis=-1) * len(ids)
                           for ids in np.array_split(np.arange(a.shape[2]), 4)], axis=-1)
        normalized = np.log1p(groups / noise)
        values.extend((np.median(normalized, axis=-1), np.quantile(normalized, 0.9, axis=-1)))
    return np.stack(values, axis=-1).astype(np.float32)


def extract_enhanced(cache, destination, protocol):
    """Cache only new observed features; original HDF5 is read-only."""
    destination = Path(destination)
    data, truth, meta = load_cache(cache)
    with h5py.File(cache, "r") as f:
        if 'amplitude' in f['windows']:
            a = f['windows/amplitude']
        else:
            # The original B2-only demo saved raw observations without CNN windows.
            observations = {link:{k:v[...] for k,v in group.items()} for link,group in f['observations'].items()}
            rebuilt = extract_windows(observations, meta['config'], meta['scene']['duration'], store_amplitude=True)
            np.testing.assert_allclose(rebuilt['time'], data['time'])
            a = rebuilt['amplitude']
        features = []
        for start in range(0, len(a), 128):
            features.append(multiscale_features(a[start:start + 128], data['calibration_noise'],
                                                meta['config']['radio']['packet_rate_hz'], protocol['feature_windows_s']))
    new = np.concatenate(features)
    features = np.concatenate([data['features'], new], axis=-1)
    assert np.isfinite(features[data['valid']]).all()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, features=features)
    return {**data, 'enhanced': features}, truth, meta


def load_enhanced(cache, destination, protocol):
    destination = Path(destination)
    if destination.exists():
        data, truth, meta = load_cache(cache)
        with np.load(destination) as f:
            data['enhanced'] = f['features']
        return data, truth, meta
    return extract_enhanced(cache, destination, protocol)


def stable_scores(model, data):
    valid = data['valid']
    result = np.full(valid.shape, np.nan)
    result[valid] = np.logaddexp(0.0, model.decision_function(data['enhanced'][valid]))
    return result


def combine(scores, data, topology, rule, cfg):
    if rule == 'quality_mean':
        return fuse(scores, data['valid'], data['quality'], topology,
                    cfg['detector']['weight_ratio_limit'], 'negative_log_static_probability')
    if rule != 'max':
        raise ValueError(rule)
    indices = TOPOLOGIES[topology]
    good = data['valid'][:, indices] & np.isfinite(scores[:, indices])
    count = good.sum(axis=1)
    out = np.max(np.where(good, scores[:, indices], -np.inf), axis=1)
    out[count == 0] = np.nan
    return out, count


def hysteresis_states(scores, on_threshold, off_threshold, on_count=1, off_count=1):
    """Separate confirmation counts for entering motion and returning to still."""
    scalar = np.ndim(on_threshold) == 0
    on, off = np.atleast_1d(on_threshold), np.atleast_1d(off_threshold)
    state = np.zeros(len(on), dtype=np.int8)
    pending = np.full(len(on), -1, dtype=np.int8)
    streak = np.zeros(len(on), dtype=np.int16)
    output = np.full((len(scores), len(on)), -1, dtype=np.int8)
    for i, score in enumerate(scores):
        if not np.isfinite(score):
            state[:], pending[:], streak[:] = -1, -1, 0
            continue
        candidate = (score >= np.where(state == 1, off, on)).astype(np.int8)
        streak = np.where(candidate == pending, streak + 1, 1)
        pending = candidate
        state = np.where(streak >= np.where(candidate == 1, on_count, off_count), candidate, state)
        output[i] = state
    return output[:, 0] if scalar else output


def select_operating_point(sequences, protocol, fpr_limit=None, threshold_only=False):
    finite = np.concatenate([s['score'][np.isfinite(s['score'])] for s in sequences])
    candidates = np.unique(np.r_[np.nextafter(finite.min(), -np.inf),
                                 np.quantile(finite, np.linspace(0, 1, protocol['threshold_quantiles'])),
                                 np.nextafter(finite.max(), np.inf)])
    fpr_limit = protocol['overall_fpr_limit'] if fpr_limit is None else fpr_limit
    settings = [(2, 2, 1.0)] if threshold_only else [(a, b, ratio) for a, b in protocol['confirmation_candidates'] for ratio in protocol['off_threshold_ratios']]
    best = None
    for on_count, off_count, ratio in settings:
        totals = {key: np.zeros(len(candidates), dtype=np.int64) for key in ('tp', 'fp', 'fn', 'tn', 'steady_fp', 'steady_n', 'walk_tp', 'walk_n', 'wave_tp', 'wave_n')}
        for seq in sequences:
            state = hysteresis_states(seq['score'], candidates, candidates * ratio, on_count, off_count)
            y = seq['truth']['label'][:, None]
            known = state >= 0
            totals['tp'] += ((state == 1) & (y == 1)).sum(axis=0)
            totals['fp'] += ((state == 1) & (y == 0)).sum(axis=0)
            totals['fn'] += ((state == 0) & (y == 1)).sum(axis=0)
            totals['tn'] += ((state == 0) & (y == 0)).sum(axis=0)
            steady = (seq['truth']['motion_fraction'] <= 1e-9)[:, None] & (y == 0)
            totals['steady_fp'] += ((state == 1) & steady).sum(axis=0)
            totals['steady_n'] += (known & steady).sum(axis=0)
            for kind in ('walk', 'wave'):
                mask = (seq['truth']['kind'] == kind)[:, None]
                totals[f'{kind}_tp'] += ((state == 1) & mask).sum(axis=0)
                totals[f'{kind}_n'] += (known & mask).sum(axis=0)
        f1 = 2 * totals['tp'] / np.maximum(2 * totals['tp'] + totals['fp'] + totals['fn'], 1)
        fpr = totals['fp'] / np.maximum(totals['fp'] + totals['tn'], 1)
        steady_fpr = totals['steady_fp'] / np.maximum(totals['steady_n'], 1)
        allowed = np.flatnonzero((fpr <= fpr_limit + 1e-12) & (steady_fpr <= protocol['steady_static_fpr_limit'] + 1e-12))
        i = max(allowed, key=lambda k: (f1[k], -fpr[k], candidates[k]))
        row = {'threshold':float(candidates[i]), 'off_threshold':float(candidates[i] * ratio),
               'on_count':on_count, 'off_count':off_count, 'validation_f1':float(f1[i]),
               'validation_fpr':float(fpr[i]), 'validation_steady_fpr':float(steady_fpr[i]),
               'validation_recall':float(totals['tp'][i] / max(totals['tp'][i] + totals['fn'][i], 1)),
               'validation_walk_recall':float(totals['walk_tp'][i] / max(totals['walk_n'][i], 1)),
               'validation_wave_recall':float(totals['wave_tp'][i] / max(totals['wave_n'][i], 1)),
               'fpr_limit':fpr_limit, 'steady_fpr_limit':protocol['steady_static_fpr_limit'], 'candidate_count':len(candidates)}
        if best is None or (row['validation_f1'], -row['validation_fpr']) > (best['validation_f1'], -best['validation_fpr']):
            best = row
    return best


def fit_and_select(train, validation, cfg, protocol, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([d['enhanced'][d['valid']] for d, _, _ in train])
    y = np.concatenate([np.broadcast_to(t['label'][:, None], d['valid'].shape)[d['valid']] for d, t, _ in train])
    kind = np.concatenate([np.broadcast_to(t['kind'][:, None], d['valid'].shape)[d['valid']] for d, t, _ in train])
    weight = np.zeros(len(y))
    for k, mass in protocol['action_weight'].items():
        indices = kind == k
        weight[indices] = mass / indices.sum()
    weight *= len(weight)
    candidate_rows, fitted = [], {}
    for name in protocol['candidates']:
        if name == 'logistic_multiscale':
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            model.fit(x, y, logisticregression__sample_weight=weight)
        else:
            model = HistGradientBoostingClassifier(max_leaf_nodes=int(name[3:]), max_iter=protocol['hgb_iterations'],
                        learning_rate=protocol['hgb_learning_rate'], min_samples_leaf=protocol['hgb_min_samples_leaf'],
                        l2_regularization=protocol['hgb_l2'], early_stopping=False, random_state=cfg['seed'])
            model.fit(x, y, sample_weight=weight)
        fitted[name] = model
        joblib.dump(model, folder / f'{name}.joblib')
        scores = [stable_scores(model, d) for d, _, _ in validation]
        for rule in protocol['fusion_candidates']:
            sequences = [{'score':combine(s, d, 'four_quality', rule, cfg)[0], 'truth':t}
                         for s, (d, t, _) in zip(scores, validation)]
            row = {'candidate':name, 'fusion':rule, **select_operating_point(sequences, protocol)}
            candidate_rows.append(row)
            print(f"  candidate {name}/{rule}: validation F1={row['validation_f1']:.4f}, wave recall={row['validation_wave_recall']:.4f}, FPR={row['validation_fpr']:.4f}", flush=True)
    selected = max(candidate_rows, key=lambda r:(r['validation_f1'], -r['validation_fpr']))
    model = fitted[selected['candidate']]
    joblib.dump(model, folder / 'selected.joblib')
    write_csv(folder / 'candidate_validation.csv', candidate_rows)
    write_json(folder / 'selection.json', {'selected':selected, 'examples':len(y), 'features':x.shape[1],
                'target':'current decision-time motion label, including all transition windows',
                'action_example_counts':{k:int((kind == k).sum()) for k in protocol['action_weight']},
                'feature_names': FEATURE_NAMES + [f'{sec:g}s_{stat}' for sec in protocol['feature_windows_s'] for stat in MULTISCALE_STATS]})
    return model, selected
