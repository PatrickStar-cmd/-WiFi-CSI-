"""Calibrated, zone-bound detection from shared CSI amplitude observations."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .geometry import Channel3D, arm_joints


SEED = 276041
WINDOW = 1.2
SAMPLES_PER_CLASS = 320
MIN_LINKS = 2
CLASSES = ('00', '10', '01', '11')


def _select(record, keys):
    return {key: record[key] for key in keys if key in record}


def geometry_config(project):
    """Copy the public layout and calibration settings, excluding scenario truth."""
    return {
        'room': list(project['room']),
        'radio': _select(project['radio'], ('frequencyGHz', 'bandwidthMHz', 'rate', 'snr', 'bodyDb', 'loss')),
        'nodes': [{**_select(node, ('id', 'role', 'antennas', 'heading')), 'position': list(node['position'])}
                  for node in project['nodes']],
        'walls': [{**_select(wall, ('id', 'height', 'thickness', 'enabled', 'lossDb', 'reflection')),
                   'start': list(wall['start']), 'end': list(wall['end'])} for wall in project.get('walls', [])],
        'furniture': [{**_select(item, ('id', 'kind', 'heading', 'enabled', 'lossDb', 'reflection')),
                       'position': list(item['position']), 'size': list(item['size'])} for item in project['furniture']],
        'wallReflection': project['wallReflection'],
        'zones': [list(person['zone']) for person in project['people']],
    }


@dataclass(frozen=True)
class Calibration:
    model: HistGradientBoostingClassifier
    names: tuple
    rate: int
    confidence_threshold: float
    metadata: dict


def _body_points(rng, zone, motions, offsets, obstacles):
    """Sample independent walking, waving and stationary calibration windows."""
    count = len(motions)
    low, high = np.asarray(zone, dtype=float).reshape((2, 2))
    yaw = rng.uniform(-np.pi, np.pi, count)
    forward = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros(count)))
    side = np.column_stack((-np.sin(yaw), np.cos(yaw), np.zeros(count)))
    walking, waving = motions == 1, motions == 2
    speed = rng.uniform(.13, 1.25, count) * walking
    span = np.ptp(offsets)
    delta = forward[:, :2] * speed[:, None] * span
    limit = np.max(np.abs(delta) / (high - low), axis=1)
    delta /= np.maximum(limit / .8, 1)[:, None]
    half = np.abs(delta) / 2
    center = rng.uniform(low + half, high - half)

    # Avoid sampling targets whose walking path would intersect solid furniture or walls.
    for _ in range(12):
        start = np.column_stack((center - delta / 2, np.full(count, 1.1)))
        end = np.column_stack((center + delta / 2, np.full(count, 1.1)))
        blocked = np.zeros(count, dtype=bool)
        for obstacle in obstacles:
            blocked |= obstacle.crosses(start, end) | obstacle.contains(start) | obstacle.contains(end)
        if not np.any(blocked):
            break
        center[blocked] = rng.uniform((low + half)[blocked], (high - half)[blocked])

    xy = center[:, None, :] + delta[:, None, :] * (offsets[None, :, None] / span)
    root = np.concatenate((xy, np.zeros((*xy.shape[:2], 1))), axis=2)
    forward = forward[:, None, :]
    side = side[:, None, :]
    step = 2 * np.pi * rng.uniform(1.15, 1.8, count)[:, None] * offsets + rng.uniform(0, 2*np.pi, (count, 1))
    swing = (.12 * walking[:, None] * np.sin(step))[:, :, None]
    wave = 2 * np.pi * rng.uniform(.8, 2.5, count)[:, None] * offsets + rng.uniform(0, 2*np.pi, (count, 1))
    displacement = (rng.uniform(.13, .42, (count, 1)) * waving[:, None] * np.sin(wave))[:, :, None]
    lift = waving[:, None, None]
    left = .25 * side + [0, 0, .85] + swing * forward
    right = -.25 * side + [0, 0, .85] - swing * forward
    right += .4 * lift * forward + lift * [0, 0, .75] + displacement * side
    points = [root + [0, 0, 1.1]]
    for sign, target in ((1, left), (-1, right)):
        shoulder = np.broadcast_to(.205 * sign * side + [0, 0, 1.36], root.shape)
        pole = np.broadcast_to(.7 * forward + sign * side + [0, 0, -.3], root.shape)
        _, hand = arm_joints(shoulder.reshape(-1, 3), target.reshape(-1, 3), pole.reshape(-1, 3))
        points.append(root + hand.reshape(root.shape))
    return np.stack(points, axis=2)


def _echo(channel, link, points, strength):
    distance, amplitude = channel.point_paths(link, points, np.array([1, .35, .35]))
    if channel.obstacles:
        tx, rx = (channel.nodes[key]['position'] for key in link)
        amplitude *= channel._transmission(tx, points) * channel._transmission(points, rx)
    return channel._sum(distance, amplitude * channel.body_scale * strength[:, None, None])


def _link_features(amplitude, rate, floor):
    """Use short causal windows and preserve subcarrier groups on each link."""
    groups = amplitude.reshape(*amplitude.shape[:-1], 4, 14).mean(axis=-1)
    smooth = uniform_filter1d(groups, size=max(3, round(.04 * rate)), axis=1, mode='nearest')
    output = []
    floor = np.maximum(np.asarray(floor), 1e-12)
    for seconds in (.32, .64, WINDOW):
        data = smooth[:, -round(seconds * rate):]
        centered = data - data.mean(axis=1, keepdims=True)
        power = np.mean(centered**2, axis=1)
        output.append(np.log1p(14 * power / floor[..., None]))
        output.append(np.log1p(14 * np.mean(np.diff(data, axis=1)**2, axis=1) / floor[..., None]))
        total = np.abs(np.fft.rfft(centered * np.hanning(data.shape[1])[None, :, None], axis=1))**2
        frequencies = np.fft.rfftfreq(data.shape[1], 1/rate)
        total = total.mean(axis=2)
        for lo, hi in ((.5, 3.), (3., 9.), (9., 22.)):
            band = total[:, (frequencies >= lo) & (frequencies < hi)].sum(axis=1) / data.shape[1]**2
            output.append(np.log1p(14 * band / floor)[..., None])
    return np.concatenate(output, axis=1).astype(np.float32)


def _calibration_features(config):
    rng = np.random.default_rng(SEED)
    rate = config['radio']['rate']
    labels = np.repeat(np.arange(4), SAMPLES_PER_CLASS)
    rng.shuffle(labels)
    offsets = (np.arange(round(WINDOW * rate)) - (round(WINDOW * rate) - 1) / 2) / rate
    environments = [Channel3D({**config, 'seed': SEED + i, 'present': False}) for i in range(4)]
    features = []
    for start in range(0, len(labels), 40):
        channel = environments[(start // 40) % len(environments)]
        kinds = labels[start:start+40]
        bits = np.column_stack((kinds & 1, kinds >> 1))
        motions = bits * (1 + (rng.random(bits.shape) < .42))
        # Include quieter and stronger echoes without using evaluation identities.
        strengths = rng.uniform(.85, 1.15, bits.shape)
        points = [_body_points(rng, zone, motions[:, i], offsets, channel.obstacles)
                  for i, zone in enumerate(config['zones'])]
        rows = []
        for link in channel.links:
            h = channel.background[link] + sum(_echo(channel, link, pts, strengths[:, i])
                                               for i, pts in enumerate(points))
            floor = channel.noise_power[link] / 2
            noise = np.sqrt(floor) * (rng.standard_normal(h.shape) + 1j*rng.standard_normal(h.shape))
            # Circular complex noise is invariant to the receiver's phase rotation in magnitude.
            link_features = _link_features(np.abs(h + noise), rate, floor)
            # Random link unavailability during calibration gives the classifier a missing input case.
            link_features[rng.random(len(kinds)) < .05] = np.nan
            rows.append(link_features)
        features.append(np.concatenate(rows, axis=1))
    return np.concatenate(features), labels, tuple(environments[0].links)


@lru_cache(maxsize=4)
def _fit(key):
    config = json.loads(key)
    training, labels, names = _calibration_features(config)
    with threadpool_limits(limits=2):
        model = HistGradientBoostingClassifier(max_iter=115, max_leaf_nodes=16, min_samples_leaf=10,
                                               learning_rate=.1, l2_regularization=.6, random_state=SEED)
        model.fit(training, labels)
    threshold = .53
    metadata = {
        'method': 'zone-bound mixed-amplitude joint HistGradientBoosting',
        'geometryHash': hashlib.sha256(key.encode()).hexdigest(),
        'calibrationSeed': SEED, 'windowsPerCombination': SAMPLES_PER_CLASS,
        'windowSeconds': WINDOW, 'classOrder': list(CLASSES),
        'confidenceThreshold': threshold, 'minimumValidLinks': MIN_LINKS,
    }
    return Calibration(model, names, config['radio']['rate'], threshold, metadata)


def calibrate(config, progress=None):
    """Train (or retrieve) a model from public geometry and unrelated synthetic windows."""
    if set(config) != {'room', 'radio', 'nodes', 'walls', 'furniture', 'wallReflection', 'zones'}:
        raise ValueError('双人校准只接受公开几何和活动区。')
    if progress:
        progress(0, '双人活动区校准')
    calibration = _fit(json.dumps(config, sort_keys=True, separators=(',', ':')))
    if progress:
        progress(100, '双人检测器就绪')
    return calibration


def predict(observations, rate, names, times, calibration):
    """Predict the four joint states from observations, with no scenario truth input."""
    if tuple(names) != calibration.names or rate != calibration.rate:
        raise ValueError('链路或采样率与双人校准不一致。')
    count = len(times)
    valid_links = np.zeros((count, len(names)), dtype=bool)
    samples = round(WINDOW * rate)
    offsets = np.arange(samples-1, -1, -1) / rate
    rows = []
    for i, link in enumerate(names):
        record = observations[link]
        usable = record['valid'] & np.isfinite(record['csi']).all(axis=1)
        stamp = record['time'][usable]
        amplitude = np.abs(record['csi'][usable])
        windows = np.zeros((count, samples, 56), dtype=np.float32)
        quiet = amplitude[stamp < 0]
        if len(quiet) >= max(3, round(rate/2)):
            # Lag differences are robust to very slow baseline drift during the static lead-in.
            floor = np.median(np.mean(np.diff(quiet, axis=0)**2, axis=0)) / 2
            for j, now in enumerate(times):
                first = np.searchsorted(stamp, now-WINDOW, side='right')
                last = np.searchsorted(stamp, now+1e-9, side='right')
                if last - first < .76 * samples:
                    continue
                t = stamp[first:last]
                if np.max(np.diff(np.r_[now-WINDOW, t, now])) > .16:
                    continue
                grid = now - offsets
                windows[j] = np.stack([np.interp(grid, t, col) for col in amplitude[first:last].T], axis=1)
                valid_links[j, i] = True
        else:
            floor = 1.
        features = _link_features(windows, rate, floor)
        features[~valid_links[:, i]] = np.nan
        rows.append(features)
    valid = valid_links.sum(axis=1) >= MIN_LINKS
    posterior = np.full((count, len(CLASSES)), np.nan)
    if valid.any():
        with threadpool_limits(limits=2):
            posterior[valid] = calibration.model.predict_proba(np.concatenate(rows, axis=1)[valid])
    return {'posterior': posterior, 'valid': valid, 'confidence': posterior.max(axis=1), 'valid_links': valid_links}
