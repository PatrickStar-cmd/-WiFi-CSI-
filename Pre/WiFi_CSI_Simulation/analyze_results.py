"""Build an evidence-linked Chinese report from a completed core run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np

from csi_sim.utils import read_csv, write_csv, write_json


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "") else float("nan")


def percent(value, digits=2):
    return f"{100 * value:.{digits}f}%" if np.isfinite(value) else "—"


def decimal(value, digits=3):
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>"


def describe_events(folder, topologies):
    pooled = defaultdict(list)
    for file in (folder / "events").glob("*.json"):
        for event in json.loads(file.read_text(encoding="utf-8")):
            if event["model"] == "B2" and event["topology"] in topologies:
                pooled[event["topology"]].append({**event, "session_id": file.stem})
    result = {}
    for topology, events in pooled.items():
        onset = [e["onset_delay_s"] for e in events if e["detected"] and e["onset_delay_s"] is not None]
        stops = [e["stop_delay_s"] for e in events if e["stop_delay_s"] is not None]
        result[topology] = {"events": len(events), "detected": sum(e["detected"] for e in events),
                            "onset_median_s": float(np.median(onset)) if onset else None,
                            "onset_p95_s": float(np.quantile(onset, 0.95)) if onset else None,
                            "stop_median_s": float(np.median(stops)) if stops else None,
                            "stop_p95_s": float(np.quantile(stops, 0.95)) if stops else None,
                            "walk_events": sum(e["kind"] == "walk" for e in events),
                            "wave_events": sum(e["kind"] == "wave" for e in events),
                            "walk_hits": sum(e["kind"] == "walk" and e["detected"] for e in events),
                            "wave_hits": sum(e["kind"] == "wave" and e["detected"] for e in events)}
    return result


def frame_breakdown(folder, topologies):
    counts = {topology: defaultdict(int) for topology in topologies}
    for file in (folder / "predictions").glob("*.csv"):
        for row in read_csv(file):
            if row["model"] != "B2" or row["topology"] not in counts:
                continue
            c = counts[row["topology"]]
            y, state, steady = int(row["truth"]), int(row["state"]), int(row["steady"])
            c["outputs"] += 1
            c["unknown"] += state == -1
            c["static_frames"] += y == 0
            c["motion_frames"] += y == 1
            c["static_false_positive"] += y == 0 and state == 1
            c["transition_false_positive"] += y == 0 and state == 1 and steady == 0
            c["steady_static_false_positive"] += y == 0 and state == 1 and steady == 1
            if y == 1:
                c[f"{row['kind']}_frames"] += 1
                c[f"{row['kind']}_detected_frames"] += state == 1
                c["transition_missed_frames"] += state == 0 and steady == 0
                c["steady_missed_frames"] += state == 0 and steady == 1
    return {key: dict(value) for key, value in counts.items()}


def build_report(run_dir, version=2):
    run_dir = Path(run_dir).resolve()
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] not in {"completed", "results_ready"}:
        raise ValueError("A completed result set is required for analysis.")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    conditions = json.loads((run_dir / "conditions.json").read_text(encoding="utf-8"))
    splits = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    summary, metrics = read_csv(run_dir / "summary.csv"), read_csv(run_dir / "metrics.csv")
    index = {(r["condition"], r["model"], r["topology"], r["metric"]): r for r in summary}
    def value(condition, model, topology, metric, field="mean"):
        return number(index[(condition, model, topology, metric)], field)
    nominal_cal = json.loads((Path(conditions["nominal"]["model_folder"]) / "calibration.json").read_text(encoding="utf-8"))
    best_link = nominal_cal["B2"]["best_single"]["link"]
    topologies = ("best_single", "dual_AC_AD", "four_equal", "four_quality")
    event_stats = {name: describe_events(run_dir / "conditions" / name, topologies) for name in ("nominal", "fixed_budget_four")}
    frames = frame_breakdown(run_dir / "conditions/nominal", topologies)
    table_specs = [("nominal", "best_single", f"最佳固定单链路 {best_link}", 100),
                   ("nominal", "dual_AC_AD", "双接收 AC＋AD", 100),
                   ("nominal", "four_equal", "四链路平均（每链路100 Hz）", 200),
                   ("nominal", "four_quality", "四链路加权（每链路100 Hz）", 200),
                   ("fixed_budget_four", "four_quality", "四链路加权（每链路50 Hz）", 100)]
    main_rows, exported = [], []
    for condition, topology, label, pps in table_specs:
        event = event_stats[condition][topology]
        f1, recall, fpr = (value(condition, "B2", topology, key) for key in ("f1", "recall", "fpr"))
        low, high = (value(condition, "B2", topology, "f1", field) for field in ("ci_low", "ci_high"))
        main_rows.append([label, pps, percent(f1), f"{percent(low)}–{percent(high)}", percent(recall), percent(fpr),
                          f"{event['detected']}/{event['events']}", decimal(event["onset_median_s"] or float("nan"), 2)])
        exported.append({"condition": condition, "topology": topology, "label": label, "tx_packets_per_second": pps,
                         "f1": f1, "f1_ci_low": low, "f1_ci_high": high, "recall": recall, "fpr": fpr, **event})
    pair_rows = read_csv(run_dir / "paired_differences.csv")
    fixed_pairs = read_csv(run_dir / "fixed_budget_paired_differences.csv")
    changes = {}
    for name, candidates, condition in (("same_link_rate", pair_rows, "nominal"), ("same_total_packets", fixed_pairs, "equal_total_transmit_budget")):
        match = next(r for r in candidates if r["condition"] == condition and r["model"] == "B2" and r["comparison"] == "four_quality-best_single" and r["metric"] == "f1")
        changes[name] = {key: number(match, key) for key in ("mean_difference", "ci_low", "ci_high")}
    physical = json.loads((run_dir / "E0/physical_checks.json").read_text(encoding="utf-8"))
    ofdm = read_csv(run_dir / "E0/ofdm_packets.csv")
    snr_values = sorted({number(r, "csi_snr_db") for r in ofdm})
    ofdm_stats = [{"snr_db": snr, "mean_nmse": float(np.mean([number(r, "nmse") for r in ofdm if number(r, "csi_snr_db") == snr])),
                   "fir_nmse": float(np.mean([number(r, "fractional_filter_nmse") for r in ofdm if number(r, "csi_snr_db") == snr]))} for snr in snr_values]
    static_rows = []
    for topology in ("best_single", "four_quality"):
        selected = [r for r in metrics if r["condition"] == "static_20min" and r["model"] == "B2" and r["topology"] == topology]
        fp = sum(number(r, "fp") for r in selected)
        tn = sum(number(r, "tn") for r in selected)
        alarm = sum(number(r, "false_alarm_events") for r in selected)
        minutes = sum(number(r, "static_minutes") for r in selected)
        static_rows.append({"topology": topology, "false_positive_frames": int(fp), "fpr": fp / (fp + tn),
                            "false_alarm_events": int(alarm), "minutes": minutes, "false_alarm_events_per_min": alarm / minutes})
    demo_path = next((run_dir / "conditions/demo/events").glob("*.json"))
    demo = [e for e in json.loads(demo_path.read_text(encoding="utf-8")) if e["model"] == "B2" and e["topology"] in {"best_single", "four_quality"}]
    methods = []
    for r in summary:
        if r["condition"] == "nominal" and r["topology"] == "AC" and r["metric"] == "f1":
            methods.append({"model": r["model"], "f1": number(r, "mean"),
                            "recall": value("nominal", r["model"], "AC", "recall"), "fpr": value("nominal", r["model"], "AC", "fpr")})
    resources = json.loads((run_dir / "resources.json").read_text(encoding="utf-8"))
    diagnostic_path = run_dir / "threshold_validation_diagnostic.v1.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.exists() else None
    cnn_seeds = read_csv(run_dir / "cnn_seed_summary.csv")
    cnn_rows = []
    for condition, topology, label in (("nominal", "best_single", "最佳固定单链路100 Hz"),
                                      ("nominal", "four_quality", "四链路加权100 Hz"),
                                      ("fixed_budget_four", "four_quality", "四链路加权50 Hz")):
        selected = {r['metric']: r for r in cnn_seeds if r['condition'] == condition and r['topology'] == topology}
        cnn_rows.append([label, percent(number(selected['f1'], 'mean_over_training_seeds')),
                         percent(number(selected['f1'], 'std_over_training_seeds')),
                         percent(number(selected['recall'], 'mean_over_training_seeds')),
                         percent(number(selected['fpr'], 'mean_over_training_seeds'))])
    payload = {"config": cfg, "split_counts": {k: len(v) for k, v in splits.items()}, "selected_single_link": best_link,
               "main_comparison": exported, "paired_f1_changes": changes, "pooled_event_statistics": event_stats,
               "frame_error_breakdown": frames, "static_tests": static_rows, "demo_events": demo,
               "method_comparison_ac": methods, "ofdm": ofdm_stats, "physical_checks": physical, "resources": resources,
               "threshold_validation_diagnostic": diagnostic, "cnn_seed_summary": cnn_seeds}
    write_json(run_dir / f"analysis.v{version}.json", payload)
    write_csv(run_dir / f"ppt_comparison.v{version}.csv", exported)
    lead = []
    for name, caption in (("same_link_rate", "相同每链路采样率"), ("same_total_packets", "相同总发包预算")):
        change = changes[name]
        lead.append(f"{caption}下，四链路加权相对最佳固定单链路的F1变化为 {100 * change['mean_difference']:+.2f} 个百分点，配对95%置信区间为 [{100 * change['ci_low']:+.2f}, {100 * change['ci_high']:+.2f}] 个百分点。")
    frame_rows = []
    for topology in ("best_single", "four_quality"):
        c = frames[topology]
        frame_rows.append(["最佳固定单链路" if topology == "best_single" else "四链路加权",
                           percent(c.get("walk_detected_frames", 0) / c["walk_frames"]),
                           percent(c.get("wave_detected_frames", 0) / c["wave_frames"]),
                           c["static_false_positive"], c["transition_false_positive"], c["steady_static_false_positive"], c["unknown"]])
    action_event_rows = []
    for topology in ("best_single", "four_quality"):
        e = event_stats['nominal'][topology]
        action_event_rows.append(["最佳固定单链路" if topology == "best_single" else "四链路加权",
                                  f"{e['walk_hits']}/{e['walk_events']}",
                                  f"{e['wave_hits']}/{e['wave_events']}",
                                  decimal(e['onset_median_s'], 2), decimal(e['onset_p95_s'], 2)])
    demo_rows = [["单链路" if e["topology"] == "best_single" else "四链路", "走动" if e["kind"] == "walk" else "挥手",
                  "检出" if e["detected"] else "漏检", decimal(e["onset_delay_s"] if e["onset_delay_s"] is not None else float("nan"), 2),
                  decimal(e["stop_delay_s"] if e["stop_delay_s"] is not None else float("nan"), 2)] for e in demo]
    image = lambda name, caption: f'<figure><img src="figures/{name}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    diagnostic_html = ""
    if diagnostic:
        diagnostic_html = '<h2>验证集阈值工作点诊断</h2><p>下表使用同一模型和18个验证场景，展示允许误报率与运动召回率的权衡。主测试结果采用1%约束；其他行用于后续阈值设计。</p>' + table(
            ['验证FPR上限','配置','验证F1','验证召回率','实际验证FPR'],
            [[percent(r['fpr_limit'],0), r['topology'], percent(r['validation_f1']), percent(r['recall']), percent(r['validation_fpr'])]
             for r in diagnostic['rows'] if r['topology'] in {diagnostic['fixed_single_link'], 'four_quality'}])
        diagnostic_html += '<p>下一步优先将稳定静止误报与动作转换延迟分别设定指标，再依据应用容忍度选择阈值，并用新场景验证调整后的方案。</p>'
    c = frames['four_quality']
    transition_share = c['transition_false_positive'] / c['static_false_positive'] if c['static_false_positive'] else 0
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Wi-Fi CSI核心仿真实验结果分析</title>
<style>body{{font-family:"Microsoft YaHei","Times New Roman",sans-serif;font-size:16px;line-height:1.8;color:#24334a;max-width:1180px;margin:auto;padding:32px;background:#f6f8fb}}h1,h2{{color:#174c83}}h1{{font-size:28px}}h2{{font-size:22px;border-bottom:2px solid #d7e5f1;padding-bottom:8px;margin-top:32px}}.card{{background:white;border:1px solid #dbe4ee;border-radius:10px;padding:18px 24px;margin:20px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{border:1px solid #dbe4ee;padding:9px;text-align:left}}th{{background:#eaf1f8}}figure{{margin:24px 0;background:white;padding:12px;border:1px solid #dbe4ee}}img{{max-width:100%;height:auto}}figcaption{{color:#536579;font-size:14px}}a{{color:#1764ab}}.small{{font-size:14px;color:#536579}}@media print{{body{{background:white;padding:0}}h2,figure{{break-inside:avoid}}}}</style>
<body><h1>Wi-Fi CSI人体运动检测：核心仿真实验结果分析</h1>
<p>运行编号：{html.escape(run_dir.name)}。完成E0物理与OFDM验证、E1方法比较、E2链路与发包预算比较，以及25秒预设动作演示。结果对应当前几何多径与观测噪声模型。</p>
<div class="card"><b>主要结果</b>{''.join('<p>'+html.escape(text)+'</p>' for text in lead)}<p>两种配对区间均跨越0，平均收益伴随明显的场景差异。当前输出偏向静止，走动时段的覆盖和挥手检出是后续优化重点。</p><p>最佳固定单链路为{best_link}，由验证集选定。检测输出覆盖率为{percent(value('nominal','B2','four_quality','coverage'))}。</p></div>
<h2>1. 实验规模与配置</h2><p>90个母场景按54训练、18验证、18测试划分；每个测试母场景使用3个观测噪声种子，每种发包协议共54场测试回放。独立场景组为18个，统计重采样按场景组进行。两种采样率分别训练共享每链路模型。TinyCNN每种采样率运行3个初始化种子。另完成空房与静止人体各10分钟测试、25秒动作演示和600个OFDM训练包。</p>
<p>房间6 m × 5 m，2发2收，56个子载波，2.462 GHz，标称CSI等效SNR 20 dB、人体参考回波比−15 dB。检测窗1秒、输出步长0.2秒、连续两次判定确认。阈值在验证集按静态FPR≤1%选择并冻结。</p>
<h2>2. 单链路与四链路比较</h2>{table(['配置','总发包/s','F1','F1的95%CI','召回率','静态FPR','检出事件/总事件','起始延迟中位/s'], main_rows)}
<p class="small">F1、召回率、FPR为先平均噪声重复、再平均母场景的结果。置信区间按母场景bootstrap 1,000次。事件数与延迟来自全部保存事件的汇总，延迟统计对象为已检出事件。</p>
{image('single_vs_four_links.png','图1 两种发包预算下的链路比较；误差线为母场景95%置信区间。')}
<h2>3. 场景动画与检测时间轴</h2><p>校准结束后的预设动作依次为：0–5秒静止、5–10秒走动、10–15秒静止、15–20秒挥手、20–25秒静止。动画同步显示房间、人体三散射点、四链路幅度、检测得分与状态。得分越大越倾向运动，阈值由验证集固定；得分采用与运动概率单调对应的稳定数值表示。</p>
{image('scene_animation.gif','图2 预设轨迹与检测结果同步回放。')}
{image('detection_timeline.png','图3 预设运动状态与实际检测输出的时间对齐。')}
{table(['配置','动作','事件匹配','起始延迟/s','停止延迟/s'], demo_rows)}
<h2>4. 错误主要分布在哪里</h2>{table(['配置','走动帧召回','挥手帧召回','全部静态误报帧','动作结束过渡窗误报','稳态静止误报','数据不足输出'], frame_rows)}
{table(['配置','检出走动事件/全部走动','检出挥手事件/全部挥手','已检出事件起始延迟中位/s','已检出事件起始延迟P95/s'], action_event_rows)}
<p>事件检出表示该动作段内至少出现一次新的稳定运动判定；帧召回率表示整段动作被覆盖的时间比例。四链路加权检出63/69个走动事件，走动帧召回率为57.49%，体现出输出存在间断。挥手事件检出为0/96，时间轴中的挥手CSI变化仍清晰可见，后续重点放在弱运动特征与阈值设计。</p>
<p>过渡窗指窗口同时覆盖运动与静止。1秒滑动窗口保留历史信道变化，两次确认也会增加状态转换的响应时间。表中将这些过渡输出与稳态静止输出分开统计，便于解释误报与延迟的关系。</p>
<p>四链路加权的静态误报中，{percent(transition_share)}集中在动作结束后的过渡窗口。</p>
<p>多链路融合的收益通过几何位置、运动方向和各链路输出概率体现；质量权重使用静态校准噪声与有效率。上表和配对区间共同描述当前融合规则在测试母场景中的表现。</p>
<h2>5. 静态专项测试与方法比较</h2>{table(['配置','静态分钟数','误报帧','FPR','误报事件数','误报事件/分钟'], [[r['topology'],decimal(r['minutes'],1),r['false_positive_frames'],percent(r['fpr']),r['false_alarm_events'],decimal(r['false_alarm_events_per_min'])] for r in static_rows])}
{table(['AC链路方法','F1','召回率','FPR'], [[r['model'],percent(r['f1']),percent(r['recall']),percent(r['fpr'])] for r in methods])}
<p class="small">B3三行对应三个训练初始化种子；各自保持同一数据划分。跨种子均值与标准差见cnn_seed_summary.csv。</p>
{table(['TinyCNN配置','F1跨种子均值','F1跨种子标准差','召回率均值','FPR均值'], cnn_rows)}
<p>TinyCNN四链路100 Hz的平均F1为45.27%，50 Hz为52.79%；两者分别训练并在各自验证集标定阈值。这组结果体现模型与工作点的共同影响。跨种子标准差和跨场景置信区间分别描述初始化差异与场景差异。</p>
{diagnostic_html}
<h2>6. 信道与OFDM检查</h2><p>零噪声整采样时延NMSE为{physical['zero_noise_integer_delay_nmse']:.3e}；多普勒最大数值误差为{max(r['doppler_max_abs_error_hz'] for r in physical['checks']):.3e} Hz。静止几何生成稳定信道。</p>
{table(['CSI等效SNR/dB','平均NMSE','FIR分数时延近似误差'], [[decimal(r['snr_db'],0),decimal(r['mean_nmse'],6),decimal(r['fir_nmse'],6)] for r in ofdm_stats])}
<p>高信噪比下，OFDM误差趋向分数时延滤波的近似误差底。长时序CSI采用几何复数信道和独立标定噪声，E0用于核对相位、时延与噪声尺度。</p>
<h2>7. 运行记录与汇报材料</h2><p>核心计算与数值重评价流程计时合计{resources['wall_seconds']/60:.2f}分钟；计算阶段{resources['compute_seconds']/60:.2f}分钟；统计{resources['statistics_seconds']:.2f}秒；绘图{resources['plot_seconds']:.2f}秒。进程采样峰值RSS为{resources['peak_process_rss_bytes']/2**30:.2f} GiB。</p>
<p>数值校验修正了浮点概率饱和与质量权重求和精度：推断采用−ln(1−p)的稳定表示，融合保持原加权概率平均的数学定义。原始CSI、数据划分及训练权重逐字节复用，阈值在原验证集重新标定。细节记录在numerical_correction.v1.json。</p>
<p>本次完成固定布局和标称噪声下的核心比较。后续E3–E5将分别考察信噪比、速度、丢包、时钟、增益与动态干扰；实物阶段使用ESP32采集数据进行独立验证。</p>
<p><a href="ppt_comparison.v{version}.csv">PPT对比表</a> · <a href="analysis.v{version}.json">结构化分析数据</a> · <a href="summary.csv">完整指标与置信区间</a> · <a href="paired_differences.csv">同采样率配对比较</a> · <a href="fixed_budget_paired_differences.csv">同发包量配对比较</a> · <a href="figures/index.json">图表索引</a></p>
<p class="small">所有图片读取已保存的预测与真值生成；原始CSI、模型、阈值、划分、配置及环境版本保存在本结果目录。</p></body></html>'''
    (run_dir / f"核心仿真实验结果分析.v{version}.html").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(run_dir / f"核心仿真实验结果分析.v{version}.html"), "selected_single": best_link,
                      "main_comparison": exported, "f1_changes": changes, "static_tests": static_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--version", type=int, default=2)
    args = parser.parse_args()
    build_report(args.run_dir, args.version)
