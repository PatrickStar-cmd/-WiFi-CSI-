"""Audit and visualize frozen optimization experiments from saved outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from csi_sim.plots import style, save_figure, timeline, scene_animation, raw_amplitudes, STATE_CMAP, STATE_NORM
from csi_sim.utils import read_csv, write_csv, write_json
from optimize_core import VARIANTS, verify_frozen, sha

NAMES = {'O0_original':'原方案', 'T1_threshold':'只调阈值', 'A1_algorithm_strict':'算法改进·1%约束', 'A2_optimized':'完整优化·5%约束'}
COLORS = ['#677A90','#E59A44','#5491C4','#239C88']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def num(row,key):
    return float(row[key]) if row.get(key) not in ('',None) else float('nan')


def pct(v):
    return f'{v*100:.2f}%' if v is not None and np.isfinite(v) else '—'


def dec(v):
    return f'{v:.2f}' if v is not None and np.isfinite(v) else '—'


def table(headers,rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def event_summary(root):
    groups = defaultdict(list)
    for file in root.glob('conditions/*/events/*.json'):
        condition = file.parent.parent.name
        for event in read(file):
            groups[(condition,event['model'],event['topology'])].append(event)
    result = {}
    for key,events in groups.items():
        row = {'events':len(events),'hits':sum(e['detected'] for e in events)}
        for kind in ('walk','wave'):
            subset = [e for e in events if e['kind'] == kind]
            row[f'{kind}_events'] = len(subset)
            row[f'{kind}_hits'] = sum(e['detected'] for e in subset)
        for name in ('onset','stop'):
            values = [e[f'{name}_delay_s'] for e in events if e['detected'] and e[f'{name}_delay_s'] is not None]
            row[f'{name}_median_s'] = float(np.median(values)) if values else None
            row[f'{name}_p95_s'] = float(np.quantile(values,.95)) if values else None
        result[key] = row
    return result


def paired_improvements(metrics,seed):
    groups = defaultdict(lambda:defaultdict(list))
    for row in metrics:
        if row['topology'] == 'four_quality':
            groups[(row['condition'],row['model'])][row['scene_id']].append(row)
    rng,output = np.random.default_rng(seed),[]
    for condition in ('nominal','fixed_budget_four','fresh_nominal','fresh_fixed_budget_four'):
        base = groups[(condition,'O0_original')]
        for variant in VARIANTS[1:]:
            new = groups[(condition,variant)]
            for metric in ('f1','recall','wave_recall','fpr'):
                differences = []
                for mother in sorted(base.keys() & new.keys()):
                    a = [num(r,metric) for r in base[mother]]
                    b = [num(r,metric) for r in new[mother]]
                    if np.isfinite(a).any() and np.isfinite(b).any():
                        differences.append(float(np.nanmean(b)-np.nanmean(a)))
                x = np.array(differences)
                boot = x[rng.integers(len(x),size=(1000,len(x)))].mean(axis=1)
                output.append({'condition':condition,'variant':variant,'metric':metric,'mean_difference':x.mean(),
                               'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'mother_count':len(x)})
    return output


def audit(root,metrics):
    verify_frozen(root)
    protocol = read(root/'optimization_protocol.v1.json')
    source = Path(protocol['baseline_run'])
    old = {(r['condition'],r['session_id'],r['topology']):r for r in read_csv(source/'metrics.csv') if r['model']=='B2'}
    compared,max_difference = 0,0.
    for row in metrics:
        key = (row['condition'],row['session_id'],row['topology'])
        if row['model']=='O0_original' and key in old:
            for k in ('f1','recall','fpr','tp','fp','fn','tn','detected_events'):
                a,b = num(row,k),num(old[key],k)
                if np.isfinite(a) and np.isfinite(b):
                    max_difference = max(max_difference,abs(a-b))
            compared += 1
    assert max_difference < 1e-12
    split = read(root/'split.json')
    assert not set(protocol['fresh_scene_ids']) & set(sum(split.values(),[]))
    conditions = read(root/'conditions.json')
    expected = {(s,n) for s in protocol['fresh_scene_ids'] for n in protocol['fresh_noise_seeds']}
    for name in ('fresh_nominal','fresh_fixed_budget_four'):
        found = {(int(r['scene_id']),int(r['noise_seed'])) for r in metrics if r['condition']==name}
        assert found == expected
    rows_checked = 0
    for file in root.glob('conditions/*/predictions/*.csv'):
        for row in read_csv(file):
            if row['probability']:
                assert 0 <= float(row['probability']) <= 1
                assert int(row['state']) in (-1,0,1)
                rows_checked += 1
    import h5py
    paired = 0
    for left in conditions['fresh_nominal']['cache_paths']:
        right = root/'conditions/fresh_fixed_budget_four/data'/Path(left).name
        with h5py.File(left,'r') as f, h5py.File(right,'r') as g:
            a,b = json.loads(f.attrs['metadata']),json.loads(g.attrs['metadata'])
            assert a['scene']==b['scene'] and a['scatterers']==b['scatterers']
        paired += 1
    result = {'status':'passed','frozen_artifact_hashes':'unchanged','original_baseline_rows_reproduced':compared,
              'baseline_max_metric_difference':max_difference,'fresh_scene_pairs':paired,
              'fresh_mothers_disjoint':True,'prediction_rows_checked':rows_checked}
    write_json(root/'audit.v1.json',result)
    return result


def before_after_plots(root,index,cfg):
    destination = root/'figures'
    fig,axes = plt.subplots(2,3,figsize=(13,7.2),layout='constrained')
    for row,condition in enumerate(('nominal','fresh_nominal')):
        for col,(metric,label) in enumerate((('f1','运动 F1'),('wave_recall','挥手帧召回率'),('fpr','整体静止误报率'))):
            ax=axes[row,col]
            selected=[index[(condition,v,'four_quality',metric)] for v in VARIANTS]
            values=np.array([num(r,'mean') for r in selected])*100
            low=np.array([num(r,'ci_low') for r in selected])*100
            high=np.array([num(r,'ci_high') for r in selected])*100
            ax.bar(np.arange(4),values,color=COLORS,yerr=np.array([np.maximum(values-low,0),np.maximum(high-values,0)]),capsize=3)
            ax.set_xticks(np.arange(4),['原方案','只调阈值','算法·1%约束','优化·5%约束'],rotation=18,fontsize=9)
            ax.set_ylabel(label+' / %')
            ax.set_title(('原18个测试场景' if row==0 else '新增18个测试场景')+'｜'+label,fontsize=11)
            ax.set_ylim(0,max(8,high.max()*1.35) if metric=='fpr' else 112)
            for i,v in enumerate(values):
                ax.text(i,high[i]+(.18 if metric=='fpr' else 2),f'{v:.1f}',ha='center',fontsize=10)
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('四链路优化前后对比｜均值与场景组95%置信区间',fontsize=16)
    save_figure(fig,destination/'before_after_comparison',cfg)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,(metric,label) in zip(axes,(('f1','运动 F1'),('wave_recall','挥手帧召回率'),('fpr','整体静止误报率'))):
        specs=[('nominal','best_single'),('nominal','four_quality'),('fixed_budget_four','four_quality')]
        for offset,variant,color in ((-.18,'O0_original',COLORS[0]),(.18,'A2_optimized',COLORS[3])):
            values=[100*num(index[(condition,variant,topology,metric)],'mean') for condition,topology in specs]
            ax.bar(np.arange(3)+offset,values,width=.34,color=color,label=NAMES[variant])
            for i,v in enumerate(values):
                ax.text(i+offset,v+(.16 if metric=='fpr' else .8),f'{v:.1f}',ha='center',fontsize=9)
        ax.set_xticks(np.arange(3),['单链路100 Hz\n总计100包/s','四链路100 Hz\n总计200包/s','四链路50 Hz\n总计100包/s'],fontsize=9)
        ax.set(ylabel=label+' / %',ylim=(0,12 if metric=='fpr' else 112),title=label)
        ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=10,frameon=False)
    fig.suptitle('单链路与四链路｜原方案与完整优化（原测试场景）',fontsize=15)
    save_figure(fig,destination/'single_vs_four_links',cfg)


def demo_plots(root,cfg,animation):
    condition=read(root/'conditions.json')['demo']
    cache=Path(condition['cache_paths'][0])
    saved=root/'conditions/demo/predictions'/f'{cache.stem}.csv'
    rows=read_csv(saved)
    # Presentation adapter explicitly selects A2 while preserving its predictions.
    adapter=root/'presentation/demo_A2.csv'
    write_csv(adapter,[{**r,'model':'B2'} for r in rows if r['model']=='A2_optimized'])
    destination=root/'figures'
    timeline(cache,adapter,destination,cfg)
    if animation:
        scene_animation(cache,adapter,destination,cfg)
    selected={v:sorted([r for r in rows if r['model']==v and r['topology']=='four_quality'],key=lambda r:float(r['time'])) for v in VARIANTS}
    cal=cfg['dataset']['calibration_s']
    stamp=np.array([num(r,'time') for r in selected['O0_original']])-cal
    fig,axes=plt.subplots(3,1,figsize=(12.4,8.2),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.3,1.1,1.5]})
    for link,(t,a) in raw_amplitudes(cache,cfg).items():
        keep=t>=cal
        axes[0].plot(t[keep]-cal,a[keep].mean(axis=1),lw=.8,label=link)
    axes[0].set_ylabel('归一化 CSI 幅度');axes[0].legend(ncol=4,loc='upper right',fontsize=9)
    for i,v in enumerate(VARIANTS):
        score=np.array([num(r,'score')/num(r,'threshold') for r in selected[v]])
        axes[1].plot(stamp,score,label=NAMES[v],color=COLORS[i],lw=1.5)
    axes[1].axhline(1,color='#444',ls=':',lw=1)
    axes[1].set_ylabel('得分 / 开启阈值');axes[1].legend(ncol=2,fontsize=9,loc='upper right')
    truth=np.array([num(r,'truth') for r in selected['O0_original']])
    matrix=np.vstack([truth]+[np.array([num(r,'state') for r in selected[v]]) for v in VARIANTS])
    axes[2].imshow(matrix,aspect='auto',interpolation='nearest',cmap=STATE_CMAP,norm=STATE_NORM,extent=(0,25,4.5,-.5))
    axes[2].set_yticks(range(5),['预设状态']+[NAMES[v] for v in VARIANTS],fontsize=10)
    axes[2].set(xlabel='校准结束后的时间 / s',xlim=(0,25))
    axes[2].legend(handles=[Line2D([0],[0],lw=7,color=c,label=l) for c,l in [('#C7DEF1','静止'),('#E58A47','运动')]],loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False)
    for start,end,label in ((5,10,'走动'),(15,20,'挥手')):
        for ax in axes[:2]:ax.axvspan(start,end,color='#E58A47',alpha=.08)
        axes[0].text((start+end)/2,1.02,label,transform=axes[0].get_xaxis_transform(),ha='center')
    fig.suptitle('优化前后检测时间轴｜同一25秒场景与同一原始 CSI',fontsize=16)
    save_figure(fig,destination/'before_after_timeline',cfg)


def report(root,animation=True):
    root=Path(root).resolve()
    if read(root/'run_status.json')['status'] not in ('evaluated','completed'):
        raise ValueError('Finish evaluation before reporting.')
    cfg=read(root/'config.json');protocol=read(root/'optimization_protocol.v1.json')
    summary,metrics=read_csv(root/'summary.csv'),read_csv(root/'metrics.csv')
    index={(r['condition'],r['model'],r['topology'],r['metric']):r for r in summary}
    def value(condition,variant,topology,metric,field='mean'):
        return num(index[(condition,variant,topology,metric)],field)
    events=event_summary(root)
    changes=paired_improvements(metrics,cfg['seed']+6)
    write_csv(root/'optimization_paired_changes.v1.csv',changes)
    exported,main_tables=[],[]
    for condition in ('nominal','fresh_nominal'):
        displayed=[]
        for variant in VARIANTS:
            row={'condition':condition,'variant':variant,'label':NAMES[variant]}
            row.update({m:value(condition,variant,'four_quality',m) for m in ('f1','recall','walk_recall','wave_recall','fpr','coverage')})
            row['f1_ci_low']=value(condition,variant,'four_quality','f1','ci_low')
            row['f1_ci_high']=value(condition,variant,'four_quality','f1','ci_high')
            row.update(events[(condition,variant,'four_quality')])
            by_mother=defaultdict(list)
            for r in metrics:
                if (r['condition'],r['model'],r['topology'])==(condition,variant,'four_quality'):
                    by_mother[r['scene_id']].append(num(r,'steady_static_fpr'))
            row['steady_static_fpr']=float(np.mean([np.mean(x) for x in by_mother.values()]))
            exported.append(row)
            displayed.append([NAMES[variant],pct(row['f1']),pct(row['recall']),pct(row['wave_recall']),pct(row['fpr']),pct(row['steady_static_fpr']),f"{row['wave_hits']}/{row['wave_events']}",dec(row['onset_median_s']),dec(row['stop_median_s'])])
        main_tables.append(table(['方案','F1','运动召回','挥手帧召回','整体静止FPR','稳态静止FPR','挥手事件检出','起始延迟中位/s','停止延迟中位/s'],displayed))
    write_csv(root/'optimization_comparison.v1.csv',exported)
    budget=[]
    for condition,topology,label in [('nominal','best_single','单链路100 Hz，总计100包/s'),('nominal','four_quality','四链路100 Hz，总计200包/s'),('fixed_budget_four','four_quality','四链路50 Hz，总计100包/s'),('fresh_fixed_budget_four','four_quality','新增场景：四链路50 Hz')]:
        for variant in ('O0_original','A2_optimized'):
            budget.append([label,NAMES[variant]]+[pct(value(condition,variant,topology,m)) for m in ('f1','recall','wave_recall','fpr')])
    static=[]
    for variant in VARIANTS:
        sel=[r for r in metrics if r['condition']=='static_20min' and r['model']==variant and r['topology']=='four_quality']
        static.append({'variant':variant,'minutes':sum(num(r,'static_minutes') for r in sel),'false_positive_frames':int(sum(num(r,'fp') for r in sel)),
                       'false_alarm_events':int(sum(num(r,'false_alarm_events') for r in sel))})
    findings=[]
    for rate in (100,50):
        folder=root/'models'/f'rate{rate}'
        findings.append({'rate':rate,'selection':read(folder/'selection.json'),'calibration':read(folder/'calibration.json'),'diagnosis':read(folder/'diagnosis.json')})
    audit_result=audit(root,metrics)
    render_start=time.perf_counter()
    style(cfg)
    before_after_plots(root,index,cfg)
    demo_plots(root,cfg,animation)
    render_seconds=time.perf_counter()-render_start
    diagnostic=findings[0]['diagnosis']
    demo=events[('demo','A2_optimized','four_quality')]
    payload={'main_comparison':exported,'paired_improvements':changes,'static_tests':static,'demo_optimized':demo,
             'selected_models':findings,'audit':audit_result,'fit_resources':read(root/'fit_resources.json'),
             'evaluation_resources':read(root/'evaluation_resources.json'),'render_seconds':render_seconds}
    write_json(root/'analysis.v3.json',payload)
    img=lambda name,caption:f'<figure><img src="figures/{name}"><figcaption>{caption}</figcaption></figure>'
    pair_table=table(['场景','对比','F1变化/百分点','配对95%CI/百分点'],[[{'nominal':'原测试场景','fresh_nominal':'新增场景'}[r['condition']],NAMES[r['variant']],f"{100*r['mean_difference']:+.2f}",f"[{100*r['ci_low']:+.2f}, {100*r['ci_high']:+.2f}]"] for r in changes if r['metric']=='f1' and r['condition'] in ('nominal','fresh_nominal')])
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Wi-Fi CSI核心仿真实验结果分析：算法优化</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:1250px;margin:auto;padding:28px;background:#f5f8fb;color:#20364c;line-height:1.8}}h1,h2{{color:#165386}}h2{{margin-top:30px;border-bottom:2px solid #cadce9}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{border:1px solid #cbd9e4;padding:8px}}th{{background:#e7f0f7}}figure{{margin:22px 0;padding:12px;background:white}}img{{max-width:100%}}figcaption,.note{{font-size:14px;color:#52677c}}.lead{{padding:18px;background:#e6f3ef;border-left:5px solid #239c88}}a{{color:#145e9c}}</style><body>
<h1>Wi-Fi CSI人体运动检测：优化前后核心实验</h1>
<div class="lead">原测试场景：四链路F1从{pct(exported[0]['f1'])}提高到{pct(exported[3]['f1'])}，运动召回率从{pct(exported[0]['recall'])}提高到{pct(exported[3]['recall'])}，挥手帧召回率从{pct(exported[0]['wave_recall'])}提高到{pct(exported[3]['wave_recall'])}。新增18个场景上的优化F1为{pct(exported[7]['f1'])}，挥手帧召回率为{pct(exported[7]['wave_recall'])}。</div>
<h2>1. 原因定位</h2>
<p><b>标签与评价时刻错位。</b>原模型按整窗运动占比产生训练标签，评价采用窗口右端状态。18个验证场景中，{diagnostic['legacy_retained_windows']}个保留窗口包含{diagnostic['legacy_endpoint_label_disagreements']}个标签与末端状态相异的窗口，另有{diagnostic['legacy_discarded_mixed_windows']}个混合窗口被筛去。动作启停附近的训练覆盖与在线判定目标因此出现差异。</p>
<p><b>整体误报约束将过渡残留转化为高阈值。</b>原1秒窗与连续两次确认保留了刚停止的运动信号。验证集四链路原阈值为{diagnostic['original_four_link_threshold']:.2f}，挥手得分中位数仅{diagnostic['four_quality_score_by_action']['wave']['median']:.2f}，90%分位数为{diagnostic['four_quality_score_by_action']['wave']['q90']:.2f}。</p>
<p><b>弱运动特征与噪声尺度。</b>模型中的挥手由单个手部散射点运动产生，躯干保持静止。原始幅度方差同时受背景信道与噪声影响。本次加入相对于静态校准噪声的变化强度、相邻采样协方差及多子载波分组统计，并使用0.2、0.4和1秒三个历史尺度。</p>
<h2>2. 优化方法和实验设置</h2>
<p>新模型输入为原16维特征与新增24维多尺度特征，共40维。使用全部有效窗口的当前时刻运动标签训练，并将静止、走动、挥手的样本权重总量设为0.5、0.25、0.25。验证比较多尺度逻辑回归、7叶和15叶直方图梯度提升树，以及加权平均／最大值融合。</p>
<p>100 Hz方案选中15叶梯度提升树＋质量加权平均；50 Hz方案选中7叶梯度提升树＋最大得分融合。分别选择开启／关闭阈值和确认次数。完整优化的验证约束为整体静止FPR≤5%、稳态静止FPR≤1%；同时保留整体FPR≤1%的算法对照。整体FPR始终包含动作转换窗口。</p>
{table(['组别','模型与特征','阈值约束与状态判定'],[['原方案','原16维特征＋原逻辑回归','原整体1% FPR标定；连续两次确认'],['只调阈值','原特征、原训练权重、原融合','整体5%＋稳态1%；保持原确认逻辑'],['算法改进·1%约束','新多尺度特征＋末端标签训练＋验证选择融合','整体1%＋稳态1%；优化状态判定'],['完整优化·5%约束','与算法改进组共享冻结的新模型','整体5%＋稳态1%；优化状态判定']])}
<p>沿用54个训练母场景与18个验证母场景。优化前后逐场回放原18个测试母场景，另在模型和阈值冻结后生成18个新母场景；每场3个噪声重复，两个发包协议合计216场运动测试回放。保留20分钟静态专项和25秒固定演示。E0复用已验证的同一信道与OFDM记录。</p>
<p class="note">全部统计来自当前几何多径模型。训练、验证、原测试、新增测试的场景ID分别保存；新增场景承担本轮独立检验。阈值与模型的冻结哈希在评价前后逐项核对。</p>
<h2>3. 原测试场景：四链路100 Hz</h2>{main_tables[0]}
<h2>4. 新增场景：四链路100 Hz</h2>{main_tables[1]}
<p class="note">帧指标先平均同一母场景的3次噪声重复，再平均母场景；事件计数汇总3次噪声重复。起始／停止延迟统计已检出的事件。稳态静止以整个原1秒窗均为静止定义，整体静止误报保留所有窗口。</p>
{pair_table}{img('before_after_comparison.png','图1 四组方案在原测试场景与新增场景上的对比。误差线按母场景bootstrap 1,000次计算。')}
<h2>5. 单链路、四链路与发包预算</h2>{table(['配置','方案','F1','运动召回','挥手帧召回','整体静止FPR'],budget)}
{img('single_vs_four_links.png','图2 每链路100 Hz和固定总发包量两种条件下的优化收益。最佳固定单链路由各方案验证集选定。')}
<h2>6. 同一场景的动画与时间轴</h2>
<p>演示动作：0–5秒静止，5–10秒走动，10–15秒静止，15–20秒挥手，20–25秒静止。各方案共享同一原始CSI；完整优化检出{demo['walk_hits']}/{demo['walk_events']}个走动事件和{demo['wave_hits']}/{demo['wave_events']}个挥手事件。</p>
{img('before_after_timeline.png','图3 同一场景的原始CSI、相对阈值得分与四组检测状态。')}
{img('scene_animation.gif','图4 完整优化方案：预设轨迹、CSI变化和检测输出同步动画。')}
<h2>7. 静态误报与下一步</h2>{table(['方案','静态分钟数','误报帧','误报事件'],[[NAMES[r['variant']],r['minutes'],r['false_positive_frames'],r['false_alarm_events']] for r in static])}
<p>本轮同时报告召回收益、整体误报、稳态误报与起停延迟。下一步使用ESP32真实CSI检查噪声标定和多尺度特征，再补充风扇、开门、丢包和跨日期场景；这些因素对应后续扰动实验。</p>
<h2>8. 可复现文件与参考</h2>
<p>训练与验证耗时{payload['fit_resources']['wall_seconds']/60:.2f}分钟；核心测试耗时{payload['evaluation_resources']['wall_seconds']/60:.2f}分钟；图表渲染{render_seconds:.1f}秒。核验复现原基线{audit_result['original_baseline_rows_reproduced']}组场次／拓扑指标，最大差异{audit_result['baseline_max_metric_difference']:.2e}。</p>
<p><a href="optimization_comparison.v1.csv">PPT对比数据表</a> · <a href="analysis.v3.json">结构化分析</a> · <a href="optimization_paired_changes.v1.csv">优化收益及配对置信区间</a> · <a href="optimization_protocol.v1.json">预设优化协议</a> · <a href="freeze.v1.json">模型与阈值冻结记录</a> · <a href="audit.v1.json">核验结果</a></p>
<p>方法参考：<a href="https://scikit-learn.org/stable/modules/classification_threshold.html">scikit-learn：分类阈值调节与独立验证</a>；<a href="https://cswu.me/papers/ubicomp19_widetect_paper.pdf">WiDetect：CSI自相关与运动统计量</a>。本项目构造噪声归一化多尺度统计特征，并通过消融实验评价。</p></body></html>'''
    (root/'核心仿真实验结果分析.v3.html').write_text(content,encoding='utf-8')
    write_json(root/'figures/index.json',{'files':[p.relative_to(root).as_posix() for p in sorted((root/'figures').iterdir()) if p.suffix in {'.png','.svg','.gif'}],
               'source':'saved predictions; scene animation explicitly selects A2_optimized'})
    write_json(root/'run_status.json',{'status':'completed','figures_rendered':True,'comparison_variants':list(VARIANTS),'source':protocol['baseline_run']})
    print(json.dumps({'report':str(root/'核心仿真实验结果分析.v3.html'),'main_comparison':exported,'static':static,'audit':audit_result},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--no-animation',action='store_true')
    args=parser.parse_args()
    report(args.run_dir,not args.no_animation)
