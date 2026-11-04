"""Explicit experiment entry points. Nothing runs at import time."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .config import changed, fingerprint
from .dataset import create_cache, load_cache, split_ids
from .evaluate import evaluate_sequence, paired_differences, summarize
from .fusion import TOPOLOGIES, confirm_states, fuse, resources
from .models import calibrate, fit_models, predict
from .utils import ResourceMonitor, environment, write_csv, write_json


class ExperimentRunner:
    def __init__(self, cfg, output):
        self.cfg, self.output = cfg, Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.splits = split_ids(cfg)
        self.bundles, self.conditions, self.metric_rows = {}, {}, []
        write_json(self.output / "config.json", cfg)
        write_json(self.output / "split.json", self.splits)
        write_json(self.output / "environment.json", environment())
        write_json(self.output / "run_status.json", {"status": "running", "config_hash": fingerprint(cfg)})

    def bundle(self, rate=None, window=None, families=None):
        rate = self.cfg["radio"]["packet_rate_hz"] if rate is None else rate
        window = self.cfg["detector"]["window_s"] if window is None else window
        wanted = [name for name in self.cfg["training"]["models"] if families is None or name in families]
        for (other_rate, other_window, _), existing in self.bundles.items():
            if other_rate == rate and other_window == window and set(wanted) <= set(existing["cfg"]["training"]["models"]):
                return existing
        key = (float(rate), float(window), "_".join(wanted))
        cfg = changed(self.cfg, {"radio.packet_rate_hz": rate, "detector.window_s": window, "training.models": wanted})
        folder = self.output / "models" / f"rate{rate:g}_window{window:g}_{key[2]}"
        folder.mkdir(parents=True)
        paths = {}
        use_cnn = "B3" in cfg["training"]["models"]
        for split in ("train", "validation"):
            paths[split] = []
            for index, mother in enumerate(self.splits[split]):
                path = folder / "data" / split / f"s{mother:06d}.h5"
                print(f"[{folder.name}] {split} {index + 1}/{len(self.splits[split])}", flush=True)
                create_cache(path, cfg, mother, split, save_raw=cfg["dataset"]["save_raw_core"], save_amplitude=use_cnn)
                paths[split].append(path)
        models, description = fit_models(paths["train"], paths["validation"], cfg, folder)
        thresholds = calibrate(models, paths["validation"], cfg, folder)
        result = {"cfg": cfg, "models": models, "description": description, "thresholds": thresholds,
                  "folder": folder, "paths": paths}
        self.bundles[key] = result
        return result

    def evaluate_condition(self, name, cfg, bundle, group, parameter="", value="", families=None, specs=None, save_raw=None, cached_paths=None):
        if name in self.conditions:
            return self.conditions[name]
        folder = self.output / "conditions" / name
        folder.mkdir(parents=True)
        families = set(families or cfg["training"]["models"])
        models = {tag: model for tag, model in bundle["models"].items() if model["kind"] in families}
        if not models:
            raise ValueError(f"No models selected for {name}")
        use_cnn = any(model["kind"] == "B3" for model in models.values())
        save_raw = cfg["dataset"]["save_raw_sweeps"] if save_raw is None else save_raw
        if specs is None:
            specs = [{"scene_id": mother, "noise_seed": noise, "mode": "random"}
                     for mother in self.splits["test"] for noise in cfg["dataset"]["test_noise_seeds"]]
        if cached_paths is not None:
            if len(cached_paths) != len(specs):
                raise ValueError("Cached sessions and specifications must have the same length.")
        condition = {"name": name, "group": group, "parameter": parameter, "value": value, "folder": str(folder),
                     "model_folder": str(bundle["folder"]), "config": cfg, "sessions": len(specs), "cache_paths": []}
        self.conditions[name] = condition
        condition_metrics = []
        for index, spec in enumerate(specs):
            mother, noise = spec["scene_id"], spec["noise_seed"]
            session_id = f"s{mother:06d}_n{noise}"
            cache = folder / "data" / f"{session_id}.h5"
            print(f"[{name}] session {index + 1}/{len(specs)} {session_id}", flush=True)
            if cached_paths is None:
                create_cache(cache, cfg, mother, "test", noise, spec.get("mode", "random"), save_raw, use_cnn,
                             spec.get("origin"), spec.get("heading"))
            else:
                import os
                cache.parent.mkdir(parents=True, exist_ok=True)
                os.link(cached_paths[index], cache)
            condition["cache_paths"].append(str(cache))
            data, truth, meta = load_cache(cache)
            rows, event_detail = [], []
            for tag, model in models.items():
                clock = time.perf_counter()
                score_matrix = predict(model, cache, data, cfg)
                prediction_seconds = time.perf_counter() - clock
                thresholds = bundle["thresholds"][tag]
                domain = "negative_log_static_probability" if model["kind"] in {"B2", "B3"} else "linear"
                for topology in thresholds:
                    selected = thresholds[topology].get("link", topology)
                    clock = time.perf_counter()
                    score, valid_links = fuse(score_matrix, data["valid"], data["quality"], selected, cfg["detector"]["weight_ratio_limit"], domain)
                    threshold = thresholds[topology]["threshold"]
                    state = confirm_states(score, threshold, cfg["detector"]["confirmation_count"])
                    fusion_seconds = time.perf_counter() - clock
                    metrics, events = evaluate_sequence(data, truth, meta, score, state)
                    indices = TOPOLOGIES[selected]
                    packet_valid = np.mean([meta["packet_valid_ratio"][("AC", "AD", "BC", "BD")[li]] for li in indices])
                    row = {"condition": name, "group": group, "parameter": parameter, "value": value,
                           "model": tag, "family": model["kind"], "model_seed": bundle["description"][tag].get("seed", 0),
                           "topology": topology, "selected_link": selected if topology == "best_single" else "",
                           "scene_id": mother, "session_id": session_id, "noise_seed": noise,
                           **metrics, **resources(selected, cfg["radio"]["packet_rate_hz"], meta["scene"]["duration"], cfg),
                           "packet_valid_ratio": packet_valid, "parameter_count": bundle["description"][tag]["parameter_count"],
                           "shared_four_link_prediction_s": prediction_seconds, "fusion_s": fusion_seconds,
                           "shared_four_link_features_s": meta["feature_seconds"],
                           "threshold": threshold, "score_domain": domain, "validation_fpr": thresholds[topology]["validation_fpr"],
                           "validation_f1": thresholds[topology]["validation_f1"]}
                    if "grid_x" in spec:
                        row.update({k: spec[k] for k in ("grid_x", "grid_y", "direction")})
                    condition_metrics.append(row)
                    event_detail.extend({"model": tag, "topology": topology, **event} for event in events)
                    for wi, stamp in enumerate(data["time"]):
                        fraction = truth["motion_fraction"][wi]
                        rows.append({"condition": name, "model": tag, "family": model["kind"], "topology": topology,
                                     "scene_id": mother, "session_id": session_id, "noise_seed": noise,
                                     "time": stamp, "score": score[wi], "threshold": threshold, "state": int(state[wi]),
                                     "score_domain": domain,
                                     "probability": -np.expm1(-score[wi]) if domain == "negative_log_static_probability" else None,
                                     "probability_threshold": -np.expm1(-threshold) if domain == "negative_log_static_probability" else None,
                                     "valid_links": int(valid_links[wi]), "truth": int(truth["label"][wi]), "kind": truth["kind"][wi],
                                     "steady": int(fraction <= 1e-9 or fraction >= 1 - 1e-9)})
            write_csv(folder / "predictions" / f"{session_id}.csv", rows)
            write_json(folder / "events" / f"{session_id}.json", event_detail)
        self.metric_rows.extend(condition_metrics)
        write_csv(folder / "metrics.csv", condition_metrics)
        write_json(folder / "condition.json", condition)
        return condition

    def nominal(self):
        return self.evaluate_condition("nominal", self.cfg, self.bundle(), "core", save_raw=self.cfg["dataset"]["save_raw_core"])

    def static_suite(self):
        specs = [{"scene_id": 900010, "noise_seed": 101, "mode": "empty"},
                 {"scene_id": 900011, "noise_seed": 101, "mode": "static"}]
        return self.evaluate_condition("static_20min", self.cfg, self.bundle(), "E1", specs=specs, save_raw=True)

    def demo(self):
        return self.evaluate_condition("demo", self.cfg, self.bundle(families=["B2"]), "demo", families=["B2"],
                                       specs=[{"scene_id": 900000, "noise_seed": 101, "mode": "demo"}], save_raw=True)

    def e2(self):
        self.nominal()
        half_rate = self.cfg["radio"]["packet_rate_hz"] / 2
        cfg = changed(self.cfg, {"radio.packet_rate_hz": half_rate})
        self.evaluate_condition("fixed_budget_four", cfg, self.bundle(rate=half_rate, families=["B2", "B3"]), "E2", "total_transmit_pps", self.cfg["radio"]["packet_rate_hz"], families=["B2", "B3"], save_raw=self.cfg["dataset"]["save_raw_core"])

    def e3(self):
        for key, config_key in (("snr_db", "observation.snr_db"), ("speed_mps", "motion.walk_speed_mps"),
                                ("body_ratio_db", "observation.body_ratio_db"), ("wave_amplitudes_m", "motion.wave_amplitude_m"),
                                ("wave_hz", "motion.wave_hz")):
            for value in self.cfg["experiments"][key]:
                cfg = changed(self.cfg, {config_key: value})
                self.evaluate_condition(f"E3_{key}_{value:g}", cfg, self.bundle(families=self.cfg["experiments"]["sweep_models"]), "E3", key, value, families=self.cfg["experiments"]["sweep_models"])

    def e4(self):
        for rate in self.cfg["experiments"]["rates_hz"]:
            cfg = changed(self.cfg, {"radio.packet_rate_hz": rate})
            self.evaluate_condition(f"E4_rate_{rate:g}", cfg, self.bundle(rate=rate, families=self.cfg["experiments"]["sweep_models"]), "E4", "rate_hz", rate, families=self.cfg["experiments"]["sweep_models"])
        for window in self.cfg["experiments"]["windows_s"]:
            cfg = changed(self.cfg, {"detector.window_s": window})
            self.evaluate_condition(f"E4_window_{window:g}", cfg, self.bundle(window=window, families=self.cfg["experiments"]["sweep_models"]), "E4", "window_s", window, families=self.cfg["experiments"]["sweep_models"])
        for loss in self.cfg["experiments"]["loss_rates"]:
            cfg = changed(self.cfg, {"observation.loss_rate": loss})
            self.evaluate_condition(f"E4_loss_{loss:g}", cfg, self.bundle(families=self.cfg["experiments"]["sweep_models"]), "E4", "loss_rate", loss, families=self.cfg["experiments"]["sweep_models"])

    def e5(self):
        variants = [("gain_drift", {"observation.gain_mode": "drift"}),
                    ("gain_step", {"observation.gain_mode": "step"}),
                    ("gain_step_uncompensated", {"observation.gain_mode": "step", "observation.gain_compensation": False}),
                    ("fan", {"observation.fan": True}),
                    ("node_c_shift", {"observation.node_c_shift_m": -0.5})]
        variants += [(f"jitter_{v:g}ms", {"observation.jitter_ms": v}) for v in self.cfg["experiments"]["jitter_ms"]]
        variants += [(f"clock_{v:g}ms", {"observation.clock_offset_ms": v}) for v in self.cfg["experiments"]["clock_ms"]]
        for duration in (1.0, 2.0, 3.0):
            variants += [(f"outage_AC_{duration:g}s", {"observation.outage_links": ["AC"], "observation.outage_duration_s": duration}),
                         (f"outage_C_{duration:g}s", {"observation.outage_links": ["AC", "BC"], "observation.outage_duration_s": duration})]
        for label, overrides in variants:
            self.evaluate_condition(f"E5_{label}", changed(self.cfg, overrides), self.bundle(families=self.cfg["experiments"]["sweep_models"]), "E5", "disturbance", label,
                                    families=self.cfg["experiments"]["sweep_models"])
        self.evaluate_condition("E5_fan_static", changed(self.cfg, {"observation.fan": True}), self.bundle(families=self.cfg["experiments"]["sweep_models"]), "E5",
                                "disturbance", "fan_static", families=self.cfg["experiments"]["sweep_models"],
                                specs=[{"scene_id": 900011, "noise_seed": 101, "mode": "static"}], save_raw=True)

    def coverage(self):
        cfg, e = self.cfg, self.cfg["experiments"]
        box, step, half = np.asarray(cfg["room"]["motion_box_m"]), e["coverage_spacing_m"], e["coverage_distance_m"] / 2
        xs = np.arange(np.ceil((box[0, 0] + half) / step) * step, box[0, 1] - half + 1e-9, step)
        ys = np.arange(np.ceil((box[1, 0] + half) / step) * step, box[1, 1] - half + 1e-9, step)
        specs = []
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                for direction in range(e["coverage_directions"]):
                    angle = 2 * np.pi * direction / e["coverage_directions"]
                    origin = np.array([x, y]) - half * np.array([np.cos(angle), np.sin(angle)])
                    mother = 1000000 + (xi * len(ys) + yi) * e["coverage_directions"] + direction
                    for noise in cfg["dataset"]["test_noise_seeds"]:
                        specs.append({"scene_id": mother, "noise_seed": noise, "mode": "coverage", "origin": origin,
                                      "heading": angle, "grid_x": x, "grid_y": y, "direction": direction})
        self.evaluate_condition("coverage", cfg, self.bundle(families=["B2"]), "E2", "spatial_grid", step, families=["B2"], specs=specs, save_raw=False)

    def finish(self):
        if self.metric_rows:
            write_csv(self.output / "metrics.csv", self.metric_rows)
            summary = summarize(self.metric_rows, self.cfg["experiments"]["bootstrap_samples"], self.cfg["seed"])
            write_csv(self.output / "summary.csv", summary)
            paired = paired_differences(self.metric_rows, self.cfg["experiments"]["bootstrap_samples"], self.cfg["seed"] + 1)
            if paired:
                write_csv(self.output / "paired_differences.csv", paired)
            self._seed_summary(summary)
            self._fixed_budget_pairs()
        write_json(self.output / "conditions.json", self.conditions)
        write_json(self.output / "run_status.json", {"status": "results_ready", "conditions": list(self.conditions),
                                                     "config_hash": fingerprint(self.cfg)})

    def _seed_summary(self, summary):
        grouped = {}
        for row in summary:
            if row["model"].startswith("B3_seed"):
                key = (row["condition"], row["topology"], row["metric"])
                grouped.setdefault(key, []).append(row["mean"])
        rows = [{"condition": k[0], "topology": k[1], "metric": k[2], "mean_over_training_seeds": np.mean(v),
                 "std_over_training_seeds": np.std(v, ddof=1) if len(v) > 1 else 0, "training_seed_count": len(v)} for k, v in grouped.items()]
        if rows:
            write_csv(self.output / "cnn_seed_summary.csv", rows)

    def _fixed_budget_pairs(self):
        if "nominal" not in self.conditions or "fixed_budget_four" not in self.conditions:
            return
        matched = []
        for row in self.metric_rows:
            keep = ((row["condition"] == "nominal" and row["topology"] == "best_single") or
                    (row["condition"] == "fixed_budget_four" and row["topology"] in {"four_equal", "four_quality"}))
            if keep:
                matched.append({**row, "condition": "equal_total_transmit_budget"})
        rows = paired_differences(matched, self.cfg["experiments"]["bootstrap_samples"], self.cfg["seed"] + 2)
        if rows:
            write_csv(self.output / "fixed_budget_paired_differences.csv", rows)


def run(cfg, output, stages, render=True):
    runner = ExperimentRunner(cfg, output)
    with ResourceMonitor() as monitor:
        compute_start = time.perf_counter()
        for stage in stages:
            print(f"Starting {stage}", flush=True)
            if stage == "E0":
                from .ofdm import run_e0
                run_e0(cfg, runner.output / "E0")
            elif stage == "E1":
                runner.nominal()
                runner.static_suite()
            elif stage == "E2":
                runner.e2()
            elif stage == "E3":
                runner.e3()
            elif stage == "E4":
                runner.e4()
            elif stage == "E5":
                runner.e5()
            elif stage == "demo":
                runner.demo()
            elif stage == "coverage":
                runner.coverage()
            elif stage == "region":
                from .region import run_region
                runner.nominal()
                run_region(runner.bundle(), runner.conditions["nominal"]["cache_paths"], cfg, runner.output / "region")
            else:
                raise ValueError(stage)
        compute_seconds = time.perf_counter() - compute_start
        report_start = time.perf_counter()
        runner.finish()
        statistics_seconds = time.perf_counter() - report_start
        plot_start = time.perf_counter()
        if render:
            from .plots import render_run
            render_run(runner.output)
        plot_seconds = time.perf_counter() - plot_start
    write_json(runner.output / "resources.json", {"wall_seconds": monitor.elapsed_s, "compute_seconds": compute_seconds,
                                                "statistics_seconds": statistics_seconds, "plot_seconds": plot_seconds,
                                                "peak_process_rss_bytes": monitor.peak_bytes})
    write_json(runner.output / "run_status.json", {"status": "completed", "conditions": list(runner.conditions),
                                                  "config_hash": fingerprint(cfg), "figures_rendered": render})
    return runner.output
