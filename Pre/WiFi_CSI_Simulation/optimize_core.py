"""Fit on the original training/validation split, freeze, then run paired tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from csi_sim.config import changed
from csi_sim.dataset import create_cache, load_cache
from csi_sim.evaluate import evaluate_sequence, summarize, paired_differences
from csi_sim.fusion import TOPOLOGIES, confirm_states, resources
from csi_sim.models import predict
from csi_sim.optimized import (load_enhanced, stable_scores, combine, hysteresis_states,
                               select_operating_point, fit_and_select)
from csi_sim.utils import ResourceMonitor, environment, read_csv, write_csv, write_json

VARIANTS = ('O0_original', 'T1_threshold', 'A1_algorithm_strict', 'A2_optimized')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_path(root, rate, group, path):
    return root / 'features' / f'rate{rate:g}' / group / (Path(path).stem + '.npz')


def original_folder(source, rate):
    return next(p for p in (source / 'models').iterdir() if p.name.startswith(f'rate{rate:g}_'))


def calibrate_variants(original, model, selection, validation, cfg, protocol):
    scores = {name: [] for name in ('old', 'new')}
    for data, _, meta in validation:
        scores['old'].append(predict(original, None, data, cfg))
        scores['new'].append(stable_scores(model, data))
    cal = {}
    for variant in VARIANTS[1:]:
        old = variant == 'T1_threshold'
        rule = 'quality_mean' if old else selection['fusion']
        limit = protocol['strict_overall_fpr_limit'] if variant == 'A1_algorithm_strict' else protocol['overall_fpr_limit']
        cal[variant] = {}
        for topology in TOPOLOGIES:
            seq = [{'score':combine(s, data, topology, rule, cfg)[0], 'truth':truth}
                   for s, (data, truth, _) in zip(scores['old' if old else 'new'], validation)]
            cal[variant][topology] = {**select_operating_point(seq, protocol, limit, threshold_only=old), 'fusion':rule}
        best = max(('AC','AD','BC','BD'), key=lambda link:(cal[variant][link]['validation_f1'], -cal[variant][link]['validation_fpr']))
        cal[variant]['best_single'] = {**cal[variant][best], 'link':best}
    return cal, scores


def diagnose(validation, baseline_scores, old_cal):
    categories = {k:[] for k in ('steady_static','stop_transition','walk','wave')}
    variance = {k:[] for k in ('still','walk','wave')}
    mismatch = retained = transitions = total = 0
    fused_category = {k:[] for k in categories}
    for (data, truth, meta), scores in zip(validation, baseline_scores):
        frac = truth['motion_fraction']
        keep = (frac <= .2 + 1e-9) | (frac >= .8 - 1e-9)
        retained += int(keep.sum())
        mismatch += int(((frac >= .8 - 1e-9).astype(int)[keep] != truth['label'][keep]).sum())
        transitions += int(((frac > .2 + 1e-9) & (frac < .8 - 1e-9)).sum())
        total += len(frac)
        masks = {'steady_static':(truth['label'] == 0) & (frac <= 1e-9),
                 'stop_transition':(truth['label'] == 0) & (frac > 1e-9),
                 'walk':truth['kind'] == 'walk', 'wave':truth['kind'] == 'wave'}
        fused = combine(scores, data, 'four_quality', 'quality_mean', meta['config'])[0]
        for key, mask in masks.items():
            categories[key].extend(scores[mask].ravel().tolist())
            fused_category[key].extend(fused[mask].tolist())
        for key in variance:
            variance[key].extend(data['features'][truth['kind'] == key,:,0].ravel().tolist())
    def quantiles(items):
        x = np.array(items); x = x[np.isfinite(x)]
        return {'n':len(x), 'q10':float(np.quantile(x,.1)), 'median':float(np.median(x)), 'q90':float(np.quantile(x,.9))}
    return {'validation_windows':total, 'legacy_retained_windows':retained, 'legacy_endpoint_label_disagreements':mismatch,
            'legacy_discarded_mixed_windows':transitions,
            'single_link_score_by_action':{k:quantiles(v) for k,v in categories.items()},
            'four_quality_score_by_action':{k:quantiles(v) for k,v in fused_category.items()},
            'raw_variance_by_action':{k:quantiles(v) for k,v in variance.items()},
            'original_four_link_threshold':old_cal['four_quality']['threshold'],
            'basis':'original validation only; no test-driven candidate selection'}


def fit(source, output, protocol_path):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg, protocol, split = read(source/'config.json'), read(protocol_path), read(source/'split.json')
    assert not set(protocol['fresh_scene_ids']) & set(sum(split.values(), []))
    write_json(output/'config.json',cfg)
    write_json(output/'optimization_protocol.v1.json',{**protocol,'frozen_before_fit_utc':datetime.now(timezone.utc).isoformat(),
               'baseline_run':str(source),'fresh_holdout_role':'new scenes generated only after model and thresholds freeze'})
    write_json(output/'split.json',split)
    write_json(output/'environment.json',environment())
    shutil.copytree(source/'E0',output/'E0',copy_function=os.link)
    write_json(output/'run_status.json',{'status':'fitting','scope':'core optimization'})
    with ResourceMonitor() as monitor, threadpool_limits(limits=2):
        for rate in (100,50):
            folder = output/'models'/f'rate{rate}'
            folder.mkdir(parents=True)
            old = original_folder(source,rate)
            protocol_cfg = changed(cfg,{'radio.packet_rate_hz':float(rate)})
            collections = {}
            for group in ('train','validation'):
                paths = sorted((old/'data'/group).glob('*.h5'))
                collections[group] = []
                for index,path in enumerate(paths):
                    if index % 9 == 0:
                        print(f'[rate{rate}] observed features {group} {index+1}/{len(paths)}',flush=True)
                    collections[group].append(load_enhanced(path,feature_path(output,rate,group,path),protocol))
            model,selected = fit_and_select(collections['train'],collections['validation'],protocol_cfg,protocol,folder)
            original = {'kind':'B2','model':joblib.load(old/'B2.joblib')}
            old_cal = read(old/'calibration.json')['B2']
            cal,scores = calibrate_variants(original,model,selected,collections['validation'],protocol_cfg,protocol)
            cal['O0_original'] = {key:{**value,'off_threshold':value['threshold'],'on_count':2,'off_count':2,'fusion':'quality_mean'} for key,value in old_cal.items()}
            joblib.dump(original['model'],folder/'original_B2.joblib')
            write_json(folder/'calibration.json',cal)
            write_json(folder/'diagnosis.json',diagnose(collections['validation'],scores['old'],old_cal))
            print(f'[rate{rate}] selected {selected["candidate"]}/{selected["fusion"]}; calibration frozen',flush=True)
            del collections
    frozen = {str(p.relative_to(output)):sha(p) for p in output.glob('models/*/*') if p.suffix in {'.json','.joblib'}}
    write_json(output/'freeze.v1.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'files':frozen,
               'protocol_sha256':sha(output/'optimization_protocol.v1.json'),'selection_data':'original 54 training and 18 validation mothers'})
    write_json(output/'fit_resources.json',{'wall_seconds':monitor.elapsed_s,'peak_process_rss_bytes':monitor.peak_bytes})
    write_json(output/'run_status.json',{'status':'frozen_ready_for_test','source':str(source)})


def verify_frozen(output):
    frozen = read(output/'freeze.v1.json')
    assert frozen['protocol_sha256'] == sha(output/'optimization_protocol.v1.json')
    for name,expected in frozen['files'].items():
        assert sha(output/name) == expected, name


def evaluate(output):
    output = output.resolve()
    verify_frozen(output)
    if read(output/'run_status.json')['status'] != 'frozen_ready_for_test':
        raise ValueError('Evaluation requires a fresh frozen result directory.')
    cfg,protocol = read(output/'config.json'),read(output/'optimization_protocol.v1.json')
    source = Path(protocol['baseline_run'])
    old_conditions = read(source/'conditions.json')
    rows,conditions = [],{}
    write_json(output/'run_status.json',{'status':'evaluating','source':str(source)})
    with ResourceMonitor() as monitor, threadpool_limits(limits=2):
        bundles = {}
        for rate in (100,50):
            folder = output/'models'/f'rate{rate}'
            bundles[rate] = {'old':{'kind':'B2','model':joblib.load(folder/'original_B2.joblib')},
                             'new':joblib.load(folder/'selected.joblib'),'cal':read(folder/'calibration.json')}
        specs = [(name,rate,'replay',old_conditions[name]['cache_paths']) for name,rate in
                 (('nominal',100),('fixed_budget_four',50),('static_20min',100),('demo',100))]
        specs += [('fresh_nominal',100,'fresh',None),('fresh_fixed_budget_four',50,'fresh',None)]
        for name,rate,role,paths in specs:
            folder = output/'conditions'/name
            protocol_cfg = changed(cfg,{'radio.packet_rate_hz':float(rate)})
            bundle = bundles[rate]
            if paths is None:
                paths = []
                for mother in protocol['fresh_scene_ids']:
                    for noise in protocol['fresh_noise_seeds']:
                        path = folder/'data'/f's{mother:06d}_n{noise}.h5'
                        create_cache(path,protocol_cfg,mother,'test',noise,save_raw=True,save_amplitude=True)
                        paths.append(str(path))
                    print(f'[{name}] generated fresh scene {mother}',flush=True)
            condition_rows = []
            conditions[name] = {'config':protocol_cfg,'cache_paths':paths,'role':role,'sessions':len(paths)}
            for index,path in enumerate(map(Path,paths)):
                print(f'[{name}] evaluate {index+1}/{len(paths)} {path.stem}',flush=True)
                data,truth,meta = load_enhanced(path,feature_path(output,rate,name,path),protocol)
                old_scores = predict(bundle['old'],path,data,protocol_cfg)
                new_scores = stable_scores(bundle['new'],data)
                predictions,events = [],[]
                for variant in VARIANTS:
                    scores = old_scores if variant in {'O0_original','T1_threshold'} else new_scores
                    for topology,cal in bundle['cal'][variant].items():
                        selected = cal.get('link',topology)
                        score,count = combine(scores,data,selected,cal['fusion'],protocol_cfg)
                        state = hysteresis_states(score,cal['threshold'],cal['off_threshold'],cal['on_count'],cal['off_count'])
                        metrics,detail = evaluate_sequence(data,truth,meta,score,state)
                        steady = (truth['label'] == 0) & (truth['motion_fraction'] <= 1e-9)
                        metrics['steady_static_fpr'] = float(np.sum((state == 1) & steady) / max(steady.sum(),1))
                        row = {'condition':name,'group':role,'model':variant,'family':variant,'topology':topology,
                               'selected_link':cal.get('link',''),'scene_id':meta['scene_id'],'noise_seed':meta['noise_seed'],
                               'session_id':meta['session_id'],**metrics,
                               **resources(selected,rate,meta['scene']['duration'],protocol_cfg),
                               'threshold':cal['threshold'],'off_threshold':cal['off_threshold'],'fusion':cal['fusion']}
                        condition_rows.append(row)
                        events.extend({'model':variant,'topology':topology,**e} for e in detail)
                        for i,stamp in enumerate(data['time']):
                            predictions.append({'model':variant,'topology':topology,'time':stamp,'score':score[i],
                                'threshold':cal['threshold'],'off_threshold':cal['off_threshold'],
                                'score_domain':'negative_log_static_probability','probability':-np.expm1(-score[i]),
                                'state':int(state[i]),'truth':int(truth['label'][i]),'kind':truth['kind'][i],
                                'steady':int(truth['motion_fraction'][i] <= 1e-9 or truth['motion_fraction'][i] >= 1-1e-9),
                                'valid_links':int(count[i]),'scene_id':meta['scene_id'],'noise_seed':meta['noise_seed']})
                write_csv(folder/'predictions'/f'{path.stem}.csv',predictions)
                write_json(folder/'events'/f'{path.stem}.json',events)
            write_csv(folder/'metrics.csv',condition_rows)
            rows.extend(condition_rows)
        write_csv(output/'metrics.csv',rows)
        write_csv(output/'summary.csv',summarize(rows,cfg['experiments']['bootstrap_samples'],cfg['seed']))
        write_csv(output/'paired_differences.csv',paired_differences(rows,cfg['experiments']['bootstrap_samples'],cfg['seed']+1))
        write_json(output/'conditions.json',conditions)
        verify_frozen(output)
    write_json(output/'evaluation_resources.json',{'wall_seconds':monitor.elapsed_s,'peak_process_rss_bytes':monitor.peak_bytes})
    write_json(output/'run_status.json',{'status':'evaluated','source':str(source),'conditions':list(conditions)})
    print(f'Core evaluation finished: {output}',flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['fit','evaluate'])
    parser.add_argument('--source',type=Path,default=Path('outputs/core_20261104_v2'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path('configs/optimization.json'))
    args = parser.parse_args()
    if args.stage == 'fit':
        fit(args.source,args.output,args.protocol)
    else:
        evaluate(args.output)
