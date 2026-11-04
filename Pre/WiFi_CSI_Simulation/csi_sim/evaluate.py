"""Frame metrics, event matching and mother-scene paired bootstrap."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score


def classification(truth, state, score):
    truth, state, score = np.asarray(truth), np.asarray(state), np.asarray(score)
    table = np.zeros((2, 3), dtype=int)
    for y in (0, 1):
        for column, prediction in enumerate((0, 1, -1)):
            table[y, column] = np.sum((truth == y) & (state == prediction))
    tn, fp, us = table[0]
    fn, tp, um = table[1]
    divide = lambda a, b: float(a / b) if b else float("nan")
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    static_f1 = divide(2 * tn, 2 * tn + fp + fn)
    finite = np.isfinite(score)
    auc = float(roc_auc_score(truth[finite], score[finite])) if np.unique(truth[finite]).size == 2 else float("nan")
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "unknown_static": int(us), "unknown_motion": int(um),
            "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn), "f1": f1,
            "macro_f1": float(np.nanmean([f1, static_f1])), "fpr": divide(fp, fp + tn), "auc": auc,
            "coverage": divide(np.sum(state >= 0), len(state)), "confusion": table.tolist()}


def event_metrics(time, truth, state, events, step_s, evaluation_end):
    time, truth, state = np.asarray(time), np.asarray(truth), np.asarray(state)
    previous = np.r_[0, state[:-1]]
    rising = (state == 1) & (previous != 1)
    records = []
    events = [event for event in events if event["end"] > time[0] and event["start"] < evaluation_end]
    for index, event in enumerate(events):
        start, end = event["start"], event["end"]
        candidates = np.flatnonzero(rising & (time >= start - 1e-9) & (time < end - 1e-9))
        before = np.flatnonzero(time < start - 1e-9)
        persistent = bool(len(before) and state[before[-1]] == 1)
        delay = float(time[candidates[0]] - start) if len(candidates) else float("nan")
        next_start = events[index + 1]["start"] if index + 1 < len(events) else evaluation_end
        stops = np.flatnonzero((time >= end - 1e-9) & (time < next_start - 1e-9) & (state == 0))
        stop_delay = float(max(0, time[stops[0]] - end)) if len(candidates) and len(stops) else float("nan")
        records.append({**event, "detected": bool(len(candidates)), "onset_delay_s": max(0.0, delay) if np.isfinite(delay) else delay,
                        "stop_delay_s": stop_delay, "alarm_at_onset": persistent})
    alarms = time[(state == 1) & (truth == 0)]
    # The interval between the end of one false-positive frame and the next start.
    fa_events = int(1 + np.sum(np.diff(alarms) - step_s >= 1.0 - 1e-9)) if len(alarms) else 0
    static_minutes = float(np.sum(truth == 0) * step_s / 60)
    onset = np.asarray([r["onset_delay_s"] for r in records if np.isfinite(r["onset_delay_s"])])
    stop = np.asarray([r["stop_delay_s"] for r in records if np.isfinite(r["stop_delay_s"])])
    summary = {"events": len(records), "detected_events": sum(r["detected"] for r in records),
               "event_recall": sum(r["detected"] for r in records) / len(records) if records else float("nan"),
               "missed_events": sum(not r["detected"] for r in records), "alarm_at_onset_events": sum(r["alarm_at_onset"] for r in records),
               "stop_unobserved_events": sum(r["detected"] and not np.isfinite(r["stop_delay_s"]) for r in records),
               "false_alarm_events": fa_events, "static_minutes": static_minutes,
               "false_alarm_events_per_min": fa_events / static_minutes if static_minutes else float("nan"),
               "unknown_seconds": float(np.sum(state < 0) * step_s)}
    for name, value in (("onset", onset), ("stop", stop)):
        summary[f"{name}_median_s"] = float(np.median(value)) if len(value) else float("nan")
        summary[f"{name}_p95_s"] = float(np.quantile(value, 0.95)) if len(value) else float("nan")
    return summary, records


def evaluate_sequence(data, truth, meta, score, state):
    result = classification(truth["label"], state, score)
    event_summary, events = event_metrics(data["time"], truth["label"], state, meta["scene"]["events"],
                                         meta["config"]["detector"]["step_s"], meta["scene"]["duration"])
    result.update(event_summary)
    steady = (truth["motion_fraction"] <= 1e-9) | (truth["motion_fraction"] >= 1 - 1e-9)
    result["steady_f1"] = classification(truth["label"][steady], state[steady], score[steady])["f1"]
    for kind in ("walk", "wave"):
        selected = truth["kind"] == kind
        denominator = np.sum(selected & (state >= 0))
        result[f"{kind}_recall"] = float(np.sum(selected & (state == 1)) / denominator) if denominator else float("nan")
        matching = [event for event in events if event["kind"] == kind]
        result[f"{kind}_event_recall"] = sum(e["detected"] for e in matching) / len(matching) if matching else float("nan")
    return result, events


SUMMARY_METRICS = ("f1", "macro_f1", "recall", "precision", "fpr", "auc", "coverage", "event_recall", "false_alarm_events_per_min",
                   "onset_median_s", "onset_p95_s", "stop_median_s", "stop_p95_s", "walk_recall", "wave_recall", "steady_f1")


def summarize(rows, bootstrap_samples, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], row["topology"])].append(row)
    rng = np.random.default_rng(seed)
    result = []
    for (condition, model, topology), group in groups.items():
        for metric in SUMMARY_METRICS:
            mothers = defaultdict(list)
            for row in group:
                value = row[metric]
                if value is not None and np.isfinite(float(value)):
                    mothers[row["scene_id"]].append(float(value))
            values = np.array([np.mean(v) for v in mothers.values()])
            if not len(values):
                continue
            samples = values[rng.integers(0, len(values), (bootstrap_samples, len(values)))].mean(axis=1)
            result.append({"condition": condition, "model": model, "topology": topology, "metric": metric,
                           "mean": float(values.mean()), "ci_low": float(np.quantile(samples, 0.025)),
                           "ci_high": float(np.quantile(samples, 0.975)), "mother_scenes": len(values), "session_repeats": len(group)})
    return result


def paired_differences(rows, bootstrap_samples, seed):
    """Four links minus the validation-selected single link, paired by mother."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row["topology"] in {"best_single", "four_equal", "four_quality"}:
            grouped[(row["condition"], row["model"])][row["topology"]][row["scene_id"]].append(row)
    rng, result = np.random.default_rng(seed), []
    for (condition, model), by_topology in grouped.items():
        for topology in ("four_equal", "four_quality"):
            ids = sorted(set(by_topology["best_single"]) & set(by_topology[topology]))
            for metric in ("f1", "recall", "fpr", "coverage", "event_recall"):
                values = []
                for mother in ids:
                    left = [r[metric] for r in by_topology[topology][mother] if np.isfinite(r[metric])]
                    right = [r[metric] for r in by_topology["best_single"][mother] if np.isfinite(r[metric])]
                    if left and right:
                        values.append(np.mean(left) - np.mean(right))
                if values:
                    values = np.asarray(values)
                    boots = values[rng.integers(len(values), size=(bootstrap_samples, len(values)))].mean(axis=1)
                    result.append({"condition": condition, "model": model, "comparison": f"{topology}-best_single", "metric": metric,
                                   "mean_difference": float(values.mean()), "ci_low": float(np.quantile(boots, 0.025)),
                                   "ci_high": float(np.quantile(boots, 0.975)), "mother_scenes": len(values)})
    return result
