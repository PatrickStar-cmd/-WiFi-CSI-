"""Publication figures and animations generated only from saved artifacts."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import roc_curve, auc

from .config import LINKS
from .dataset import load_cache
from .utils import read_csv, write_json

COLORS = ["#245CA6", "#DE8F25", "#2A9D8F", "#A564B5"]
STATE_CMAP = ListedColormap(["#BABABA", "#C7DEF1", "#E58A47"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
TOPOLOGY_LABELS = {"AC": "AC", "AD": "AD", "BC": "BC", "BD": "BD", "best_single": "最佳固定单链路", "dual_AC_AD": "双接收", "four_equal": "四链路平均", "four_quality": "四链路加权"}


def style(cfg):
    plt.rcParams.update({"font.family": cfg["plots"]["font"], "font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "savefig.facecolor": "white"})


def save_figure(fig, stem, cfg):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in cfg["plots"]["formats"]:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    plt.close(fig)


def numeric(row, key):
    return float(row[key]) if row.get(key) not in (None, "") else float("nan")


def select_predictions(rows, topology, model="B2"):
    chosen = sorted([row for row in rows if row["model"] == model and row["topology"] == topology], key=lambda row: float(row["time"]))
    if not chosen:
        raise ValueError(f"Missing saved predictions: {model}/{topology}")
    result = {key: np.array([numeric(row, key) for row in chosen]) for key in ("time", "score", "threshold", "state", "truth", "valid_links")}
    # Saved v1 files contain probability scores; v2 records the stable score domain.
    result["score_domain"] = chosen[0].get("score_domain", "probability")
    return result


def score_axis(single, four):
    if single["score_domain"] == "negative_log_static_probability":
        values = np.r_[single["score"], four["score"], single["threshold"], four["threshold"]]
        maximum = max(1.0, float(np.max(values[np.isfinite(values)])))
        return "运动检测得分", (-0.03 * maximum, 1.05 * maximum)
    return "运动概率", (-0.03, 1.03)


def raw_amplitudes(cache, cfg):
    with h5py.File(cache, "r") as f:
        scales = f["windows/calibration_rms"][...]
        result = {}
        for li, link in enumerate(LINKS):
            group = f[f"observations/{link}"]
            amplitude = abs(group["csi"][...]).astype(float)
            if cfg["observation"]["gain_compensation"]:
                amplitude /= 10 ** (group["gain_db"][...][:, None] / 20)
            result[link] = (group["timestamp"][...], amplitude / scales[li])
    return result


def draw_room(ax, meta):
    width, height = meta["config"]["room"]["size_m"]
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, lw=2.5, color="#444444"))
    nodes = {key: np.asarray(value) for key, value in meta["nodes"].items()}
    for li, link in enumerate(LINKS):
        tx, rx = nodes[link[0]], nodes[link[1]]
        ax.plot([tx[0], rx[0]], [tx[1], rx[1]], "--", color=COLORS[li], alpha=0.5, lw=1)
        middle = (tx + rx) / 2
        if link in {"AD", "BC"}:
            middle = middle + np.array([-0.45 if link == "AD" else 0.45, 0.2])
        ax.text(*middle, link, color=COLORS[li], fontsize=10)
    for name, xy in nodes.items():
        ax.scatter(*xy, s=130, marker="^" if name in "AB" else "s", color="#245CA6" if name in "AB" else "#2A9D8F", zorder=5)
        ax.text(xy[0], xy[1] + 0.23, f"{name} {'发射' if name in 'AB' else '接收'}", ha="center", fontsize=10)
    scatterers = np.asarray(meta["scatterers"])
    ax.scatter(scatterers[:, 0], scatterers[:, 1], s=60, marker="s", c="#B8A78D", alpha=0.6, label="家具等效散射点")
    ax.set(xlim=(-0.25, width + 0.25), ylim=(-0.2, height + 0.6), xlabel="x / m", ylabel="y / m", aspect="equal")
    ax.set_title("房间布局与预设人体运动")
    return nodes


def state_panel(ax, single, four, cal, end):
    matrix = np.vstack([single["truth"], single["state"], four["state"]])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=STATE_CMAP, norm=STATE_NORM,
              extent=(single["time"][0] - cal, end - cal, 2.5, -0.5))
    ax.set_yticks([0, 1, 2], ["预设状态", "单链路输出", "四链路输出"])
    ax.set_xlabel("校准结束后的时间 / s")
    ax.legend(handles=[Line2D([0], [0], lw=7, color=color, label=label) for color, label in
                       zip(["#C7DEF1", "#E58A47", "#BABABA"], ["静止", "运动", "数据不足"])],
              loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False, fontsize=9)


def timeline(cache, predictions, destination, cfg):
    data, truth, meta = load_cache(cache)
    rows = read_csv(predictions)
    single, four = select_predictions(rows, "best_single"), select_predictions(rows, "four_quality")
    cal, end = cfg["dataset"]["calibration_s"], meta["scene"]["duration"]
    raw = raw_amplitudes(cache, cfg)
    fig, axes = plt.subplots(3, 1, figsize=(12.6, 8.0), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.5, 1]}, layout="constrained")
    for li, link in enumerate(LINKS):
        t, a = raw[link]
        use = (t >= cal) & (t < end)
        # Packet loss is retained as gaps in the displayed signal.
        axes[0].plot(t[use] - cal, np.mean(a[use], axis=1), lw=0.8, color=COLORS[li], label=link, alpha=0.8)
    axes[0].set(ylabel="归一化幅度", title="人体动作、CSI变化与检测输出（仿真结果）")
    axes[0].legend(ncol=4, loc="upper right")
    for s, color, label in ((single, COLORS[0], "最佳固定单链路"), (four, COLORS[2], "四链路加权")):
        axes[1].plot(s["time"] - cal, s["score"], label=label, color=color, lw=1.6)
        axes[1].axhline(s["threshold"][0], color=color, linestyle=":", alpha=0.8, label=f"{label}阈值")
    score_label, score_limits = score_axis(single, four)
    axes[1].set(ylabel=score_label, ylim=score_limits)
    axes[1].legend(ncol=2, fontsize=9, loc="upper right")
    for event in meta["scene"]["events"]:
        for ax in axes[:2]:
            ax.axvspan(event["start"] - cal, event["end"] - cal, color="#E58A47", alpha=0.09)
        axes[0].text((event["start"] + event["end"]) / 2 - cal, 1.01,
                     "走动" if event["kind"] == "walk" else "挥手", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=10)
    state_panel(axes[2], single, four, cal, end)
    axes[2].set_xlim(0, end - cal)
    save_figure(fig, Path(destination) / "detection_timeline", cfg)


def csi_heatmap(cache, destination, cfg):
    _, _, meta = load_cache(cache)
    cal = cfg["dataset"]["calibration_s"]
    raw = raw_amplitudes(cache, cfg)
    values = np.concatenate([a[np.isfinite(a)] for _, a in raw.values()])
    lo, hi = np.quantile(values, [0.02, 0.98])
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), sharex=True, sharey=True, layout="constrained")
    for ax, link in zip(axes.flat, LINKS):
        t, a = raw[link]
        keep = t >= cal
        image = ax.imshow(a[keep].T, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis", vmin=lo, vmax=hi,
                          extent=(0, meta["scene"]["duration"] - cal, 0, 56))
        ax.set(title=f"{link} 链路", xlabel="校准结束后的时间 / s", ylabel="子载波序号（按频率排序）")
    fig.colorbar(image, ax=axes, label="归一化 CSI 幅度", shrink=0.85)
    fig.suptitle("四链路 CSI 热力图（仿真结果）")
    save_figure(fig, Path(destination) / "csi_heatmap", cfg)


def scene_animation(cache, predictions, destination, cfg):
    _, _, meta = load_cache(cache)
    rows = read_csv(predictions)
    single, four = select_predictions(rows, "best_single"), select_predictions(rows, "four_quality")
    with h5py.File(cache, "r") as f:
        truth_time = f["truth/trajectory_time"][...]
        positions = f["truth/trajectory_positions"][...]
    raw = raw_amplitudes(cache, cfg)
    cal, end = cfg["dataset"]["calibration_s"], meta["scene"]["duration"]
    fig = plt.figure(figsize=(12.2, 7.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.3], height_ratios=[1.3, 1.3, 0.8])
    left_grid = grid[:, 0].subgridspec(2, 1, height_ratios=[0.16, 1.0])
    status_ax = fig.add_subplot(left_grid[0])
    status_ax.set_axis_off()
    room, amp, probability, states = fig.add_subplot(left_grid[1]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 1])
    nodes = draw_room(room, meta)
    initial = positions[np.searchsorted(truth_time, cal)]
    body = room.scatter(initial[:, 0], initial[:, 1], s=[170, 65, 65], c=["#D05B32", "#245CA6", "#245CA6"], zorder=7)
    left_arm, = room.plot([], [], color="#D05B32", lw=3, zorder=6)
    right_arm, = room.plot([], [], color="#D05B32", lw=3, zorder=6)
    path, = room.plot([], [], color="#D05B32", alpha=0.4, lw=1)
    reflection, = room.plot([], [], ":", color="#C96A2E", lw=1.2, alpha=0.65)
    status = status_ax.text(0.03, 0.5, "", transform=status_ax.transAxes, va="center", fontsize=11,
                       bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#DDDDDD"})
    room.legend(loc="lower left", fontsize=9)
    amp_lines, amp_values = [], []
    all_amp = []
    for li, link in enumerate(LINKS):
        stamp, a = raw[link]
        mask = (stamp >= cal) & (stamp < end)
        ts, value = stamp[mask] - cal, np.mean(a[mask], axis=1)
        amp_values.append((ts, value))
        all_amp.extend(value[np.isfinite(value)])
        line, = amp.plot([], [], color=COLORS[li], lw=0.9, label=link)
        amp_lines.append(line)
    low, high = np.quantile(all_amp, [0.005, 0.995])
    span = max(high - low, 0.05)
    amp.set(xlim=(0, end - cal), ylim=(low - 0.1 * span, high + 0.1 * span), ylabel="归一化幅度", title="四链路 CSI 幅度")
    amp.legend(ncol=4, fontsize=9, loc="upper right")
    score_lines = []
    for s, color, label in ((single, COLORS[0], "单链路"), (four, COLORS[2], "四链路")):
        line, = probability.plot([], [], color=color, lw=1.8, label=label)
        score_lines.append(line)
        probability.axhline(s["threshold"][0], color=color, ls=":", alpha=0.7)
    score_label, score_limits = score_axis(single, four)
    probability.set(xlim=(0, end - cal), ylim=score_limits, ylabel=score_label)
    probability.legend(loc="upper right", fontsize=9)
    state_panel(states, single, four, cal, end)
    cursors = [ax.axvline(0, color="#555555", lw=1) for ax in (amp, probability, states)]
    fig.suptitle("Wi-Fi CSI 人体运动检测｜预设轨迹与检测结果同步回放", fontsize=15)
    labels = {-1: "数据不足", 0: "静止", 1: "运动"}

    def update(relative_time):
        now = relative_time + cal
        index = min(np.searchsorted(truth_time, now, side="right") - 1, len(truth_time) - 1)
        p = positions[index]
        body.set_offsets(p)
        left_arm.set_data(p[[0, 1], 0], p[[0, 1], 1])
        right_arm.set_data(p[[0, 2], 0], p[[0, 2], 1])
        trajectory = positions[(truth_time >= cal) & (truth_time <= now), 0]
        path.set_data(trajectory[:, 0], trajectory[:, 1])
        ray = np.vstack([nodes["A"], p[0], nodes["C"]])
        reflection.set_data(ray[:, 0], ray[:, 1])
        for line, (ts, value) in zip(amp_lines, amp_values):
            keep = ts <= relative_time
            line.set_data(ts[keep], value[keep])
        for line, s in zip(score_lines, (single, four)):
            keep = s["time"] <= now + 1e-9
            line.set_data(s["time"][keep] - cal, s["score"][keep])
        for cursor in cursors:
            cursor.set_xdata([relative_time, relative_time])
        wi = max(0, np.searchsorted(single["time"], now, side="right") - 1)
        kind = next((seg["kind"] for seg in meta["scene"]["segments"] if seg["start"] <= now < seg["end"]), "still")
        status.set_text(f"时间 {relative_time:4.1f} s    预设动作：{ {'still': '静止', 'walk': '走动', 'wave': '挥手'}[kind]}\n单链路：{labels[int(single['state'][wi])]}    四链路：{labels[int(four['state'][wi])]}")
        return [body, left_arm, right_arm, path, reflection, status, *amp_lines, *score_lines, *cursors]

    fps = cfg["plots"]["animation_fps"]
    frames = np.arange(0, end - cal, 1 / fps)
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False, cache_frame_data=False)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    animation.save(destination / "scene_animation.gif", writer=PillowWriter(fps=fps), dpi=cfg["plots"]["animation_dpi"])
    update(min(7.5, end - cal - 1 / fps))
    for extension in cfg["plots"]["formats"]:
        fig.savefig(destination / f"scene_snapshot.{extension}", dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    plt.close(fig)


def link_comparison(summary, destination, cfg):
    index = {(r["condition"], r["model"], r["topology"], r["metric"]): r for r in summary}
    protocols = ["nominal"] + (["fixed_budget_four"] if any(r["condition"] == "fixed_budget_four" for r in summary) else [])
    topologies = list(TOPOLOGY_LABELS)
    fig, axes = plt.subplots(len(protocols), 3, figsize=(15.5, 4.3 * len(protocols)), squeeze=False, layout="constrained")
    for row, protocol in enumerate(protocols):
        for col, (metric, label) in enumerate((("f1", "运动 F1"), ("recall", "运动召回率"), ("fpr", "静态误报率"))):
            ax, selected = axes[row, col], []
            for topology in topologies:
                condition = protocol if protocol == "nominal" or topology.startswith("four") else "nominal"
                key = (condition, "B2", topology, metric)
                if key in index:
                    selected.append((topology, index[key]))
            if not selected:
                ax.set_visible(False)
                continue
            values = np.array([numeric(r, "mean") for _, r in selected])
            low = np.array([numeric(r, "ci_low") for _, r in selected])
            high = np.array([numeric(r, "ci_high") for _, r in selected])
            colors = [COLORS[2] if t.startswith("four") else COLORS[0] for t, _ in selected]
            ax.bar(np.arange(len(values)), values, color=colors, alpha=0.9,
                   yerr=np.vstack([np.maximum(0, values - low), np.maximum(0, high - values)]), capsize=3)
            ax.set_xticks(np.arange(len(values)), [TOPOLOGY_LABELS[t] for t, _ in selected], rotation=35, ha="right", fontsize=9)
            ax.set_ylabel(label)
            ax.set_ylim(0, max(0.05, min(1.05, np.nanmax(high) * 1.2)) if metric == "fpr" else 1.05)
            ax.grid(axis="y", alpha=0.2)
            title = "相同每链路采样率" if protocol == "nominal" else "相同总发包预算"
            ax.set_title(f"{title}｜{label}", fontsize=11)
    fig.suptitle("单链路与四链路检测效果（B2；母场景均值与95%置信区间）", fontsize=15)
    rate = cfg["radio"]["packet_rate_hz"]
    fig.supxlabel(f"上排：单/双接收{rate:g}包/s，四链路{2 * rate:g}包/s；下排：各拓扑总计{rate:g}包/s" if len(protocols) == 2 else "每链路采样率相同；通信资源记录见 metrics.csv", fontsize=10)
    save_figure(fig, Path(destination) / "single_vs_four_links", cfg)


def methods_roc_confusion(run_dir, summary, cfg):
    destination = Path(run_dir) / "figures"
    selected = [r for r in summary if r["condition"] == "nominal" and r["topology"] == "AC" and r["metric"] == "f1"]
    if selected:
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        means = np.array([numeric(r, "mean") for r in selected])
        ax.bar(np.arange(len(selected)), means, color=COLORS[0],
               yerr=np.vstack([np.maximum(0, means - [numeric(r, "ci_low") for r in selected]),
                               np.maximum(0, [numeric(r, "ci_high") for r in selected] - means)]), capsize=4)
        ax.set_xticks(np.arange(len(selected)), [r["model"] for r in selected], rotation=20)
        ax.set(ylabel="运动 F1", ylim=(0, 1.05), title="固定 AC 链路的方法比较（仿真结果）")
        save_figure(fig, destination / "method_comparison", cfg)
    grouped, confusion = defaultdict(list), {t: np.zeros((2, 3), dtype=int) for t in ("best_single", "four_quality")}
    for file in (Path(run_dir) / "conditions/nominal/predictions").glob("*.csv"):
        for row in read_csv(file):
            score = numeric(row, "score")
            if row["topology"] == "AC" and np.isfinite(score):
                grouped[row["model"]].append((int(row["truth"]), score))
            if row["model"] == "B2" and row["topology"] in confusion:
                state = int(row["state"])
                confusion[row["topology"]][int(row["truth"]), {0: 0, 1: 1, -1: 2}[state]] += 1
    if grouped:
        fig, ax = plt.subplots(figsize=(6, 5.5), layout="constrained")
        for name, values in grouped.items():
            array = np.asarray(values)
            if len(np.unique(array[:, 0])) < 2:
                continue
            fpr, tpr, _ = roc_curve(array[:, 0], array[:, 1])
            ax.plot(fpr, tpr, label=f"{name}: AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
        ax.set(xlabel="假阳性率", ylabel="真阳性率", title="AC 链路 ROC（平滑前分数；窗口汇总）")
        ax.legend(fontsize=9)
        save_figure(fig, destination / "roc", cfg)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, (name, table) in zip(axes, confusion.items()):
            ax.imshow(table, cmap="Blues", aspect="auto")
            for y in range(2):
                for x in range(3):
                    ax.text(x, y, str(table[y, x]), ha="center", va="center", color="black")
            ax.set_xticks([0, 1, 2], ["静止", "运动", "数据不足"])
            ax.set_yticks([0, 1], ["静止", "运动"])
            ax.set(xlabel="预测状态", ylabel="预设状态", title=TOPOLOGY_LABELS[name])
        save_figure(fig, destination / "confusion_matrices", cfg)


def sweep_figures(summary, conditions, destination, cfg):
    index = {(r["condition"], r["topology"], r["metric"]): r for r in summary if r["model"] == "B2"}
    grouped = defaultdict(list)
    for name, c in conditions.items():
        if c["group"] in {"E3", "E4"}:
            grouped[c["parameter"]].append((float(c["value"]), name))
    for parameter, values in grouped.items():
        values.sort()
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for ax, metric in zip(axes.flat, ("f1", "recall", "fpr", "coverage")):
            for topology, color in (("best_single", COLORS[0]), ("four_quality", COLORS[2])):
                chosen = [(x, index[(name, topology, metric)]) for x, name in values if (name, topology, metric) in index]
                if not chosen:
                    continue
                x, y = [r[0] for r in chosen], [numeric(r[1], "mean") for r in chosen]
                ax.plot(x, y, "o-", color=color, label=TOPOLOGY_LABELS[topology])
                ax.fill_between(x, [numeric(r[1], "ci_low") for r in chosen], [numeric(r[1], "ci_high") for r in chosen], alpha=0.15, color=color)
            ax.set(xlabel=parameter, ylabel=metric)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        fig.suptitle(f"{parameter} 对检测效果的影响（仿真结果）")
        save_figure(fig, Path(destination) / f"sweep_{parameter}", cfg)


def coverage_plot(run_dir, cfg):
    path = Path(run_dir) / "conditions/coverage/metrics.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    xs, ys = sorted({numeric(r, "grid_x") for r in rows}), sorted({numeric(r, "grid_y") for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, topology in zip(axes, ("best_single", "four_quality")):
        grid = np.full((len(ys), len(xs)), np.nan)
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                values = [numeric(r, "event_recall") for r in rows if r["topology"] == topology and numeric(r, "grid_x") == x and numeric(r, "grid_y") == y]
                grid[yi, xi] = np.mean(values)
        step = cfg["experiments"]["coverage_spacing_m"]
        image = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis", extent=[xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2])
        ax.set(xlabel="x / m", ylabel="y / m", title=TOPOLOGY_LABELS[topology])
    fig.colorbar(image, ax=axes, label="短运动事件召回率（方向与噪声重复平均）", shrink=0.8)
    save_figure(fig, Path(run_dir) / "figures/spatial_coverage", cfg)


def e0_figures(run_dir, cfg):
    folder = Path(run_dir) / "E0"
    rows = read_csv(folder / "ofdm_packets.csv")
    snrs = sorted({numeric(r, "csi_snr_db") for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for metric, label in (("nmse", "相对几何真值"), ("noise_nmse_vs_fir", "相对FIR真值"), ("fractional_filter_nmse", "分数时延滤波误差")):
        means = [np.mean([numeric(r, metric) for r in rows if numeric(r, "csi_snr_db") == snr]) for snr in snrs]
        axes[0].semilogy(snrs, np.maximum(means, 1e-18), "o-", label=label)
    axes[0].set(xlabel="配置的 CSI 等效 SNR / dB", ylabel="NMSE", title="OFDM信道估计误差")
    axes[0].legend(fontsize=9)
    example_path = folder / "csi_example.npz"
    if example_path.exists():
        with np.load(example_path) as d:
            axes[1].plot(d["subcarrier"], abs(d["truth"]), label="几何真值")
            axes[1].plot(d["subcarrier"], abs(d["estimate"]), ".", label="LS估计")
        axes[1].set(xlabel="子载波索引", ylabel="CSI 幅度", title="20 dB 示例")
        axes[1].legend()
    save_figure(fig, Path(run_dir) / "figures/ofdm_validation", cfg)


def disturbance_figures(summary, conditions, destination, cfg):
    names = [name for name, value in conditions.items() if value["group"] == "E5" and name != "E5_fan_static"]
    if not names:
        return
    index = {(r["condition"], r["topology"], r["metric"]): r for r in summary if r["model"] == "B2"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    labels = [str(conditions[name]["value"]) for name in names]
    for ax, (metric, ylabel) in zip(axes.flat, (("f1", "运动F1"), ("fpr", "静态误报率"), ("coverage", "输出覆盖率"), ("event_recall", "运动事件召回率"))):
        for topology, color in (("best_single", COLORS[0]), ("four_quality", COLORS[2])):
            values = [numeric(index[(name, topology, metric)], "mean") if (name, topology, metric) in index else np.nan for name in names]
            ax.plot(np.arange(len(names)), values, "o-", color=color, label=TOPOLOGY_LABELS[topology])
        ax.set_xticks(np.arange(len(names)), labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("观测与环境扰动对检测效果的影响（仿真结果）")
    save_figure(fig, Path(destination) / "disturbance_comparison", cfg)


def region_figure(run_dir, cfg):
    file = Path(run_dir) / "region/metrics.json"
    if not file.exists():
        return
    result = json.loads(file.read_text(encoding="utf-8"))
    if "confusion" not in result["main"]:
        return
    table = np.asarray(result["main"]["confusion"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8), layout="constrained")
    ax.imshow(table, cmap="Blues")
    for y in range(3):
        for x in range(3):
            ax.text(x, y, str(table[y, x]), ha="center", va="center")
    ax.set_xticks([0, 1, 2], ["左", "中", "右"])
    ax.set_yticks([0, 1, 2], ["左", "中", "右"])
    ax.set(xlabel="预测区域", ylabel="预设区域", title=f"区域识别（仿真结果；宏F1={result['main']['macro_f1']:.3f}）")
    save_figure(fig, Path(run_dir) / "figures/region_confusion", cfg)


def render_run(run_dir, animation=True):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    conditions = json.loads((run_dir / "conditions.json").read_text(encoding="utf-8"))
    style(cfg)
    destination = run_dir / "figures"
    destination.mkdir(exist_ok=True)
    if (run_dir / "E0/ofdm_packets.csv").exists():
        e0_figures(run_dir, cfg)
    if (run_dir / "summary.csv").exists():
        summary = read_csv(run_dir / "summary.csv")
        if "nominal" in conditions:
            link_comparison(summary, destination, cfg)
            methods_roc_confusion(run_dir, summary, cfg)
        sweep_figures(summary, conditions, destination, cfg)
        disturbance_figures(summary, conditions, destination, cfg)
        coverage_plot(run_dir, cfg)
        region_figure(run_dir, cfg)
    if "demo" in conditions:
        cache = Path(conditions["demo"]["cache_paths"][0])
        predictions = cache.parent.parent / "predictions" / f"{cache.stem}.csv"
        timeline(cache, predictions, destination, cfg)
        csi_heatmap(cache, destination, cfg)
        if animation:
            scene_animation(cache, predictions, destination, cfg)
    write_json(destination / "index.json", {"files": [str(p.relative_to(run_dir)) for p in sorted(destination.iterdir()) if p.is_file() and p.name != "index.json"],
                                            "source": "saved predictions, observations and ground truth", "label": "simulation results"})
