"""Recalibrate and reevaluate stored core data with stable probability arithmetic.

The original CSI, splits and trained weights are reused byte-for-byte. No model
training or new channel realization is performed by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import joblib
import torch

from csi_sim.cnn import load_cnn
from csi_sim.config import changed
from csi_sim.dataset import load_cache
from csi_sim.models import calibrate
from csi_sim.runner import ExperimentRunner
from csi_sim.utils import ResourceMonitor, write_json


def recompute(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    old_status = json.loads((source / "run_status.json").read_text(encoding="utf-8"))
    if old_status["status"] != "completed":
        raise ValueError("Source core run must finish before reevaluation.")
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    conditions = json.loads((source / "conditions.json").read_text(encoding="utf-8"))
    old_resources = json.loads((source / "resources.json").read_text(encoding="utf-8"))
    runner = ExperimentRunner(cfg, destination)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    shutil.copytree(source / "E0", destination / "E0", copy_function=os.link)
    bundles, checkpoint_hashes = {}, {}
    with ResourceMonitor() as monitor:
        start = time.perf_counter()
        for old_folder in sorted((source / "models").iterdir()):
            folder = destination / "models" / old_folder.name
            folder.mkdir(parents=True)
            shutil.copytree(old_folder / "data", folder / "data", copy_function=os.link)
            description = json.loads((old_folder / "models.json").read_text(encoding="utf-8"))
            for file in old_folder.iterdir():
                if file.is_file() and file.name != "calibration.json":
                    os.link(file, folder / file.name)
                    if file.suffix in {".pt", ".joblib"}:
                        checkpoint_hashes[str((folder / file.name).relative_to(destination))] = hashlib.sha256(file.read_bytes()).hexdigest()
            models = {}
            for tag, details in description.items():
                if tag == "training":
                    continue
                kind = details["kind"]
                if kind == "B2":
                    models[tag] = {"kind": kind, "model": joblib.load(folder / "B2.joblib")}
                elif kind == "B3":
                    models[tag] = {"kind": kind, "artifact": load_cnn(folder / f"{tag}.pt", cfg["training"]["device"])}
                else:
                    models[tag] = {"kind": kind}
            paths = {name: sorted((folder / "data" / name).glob("*.h5")) for name in ("train", "validation")}
            _, _, first_meta = load_cache(paths["validation"][0])
            families = list(dict.fromkeys(details["kind"] for tag, details in description.items() if tag != "training"))
            protocol_cfg = changed(cfg, {"radio.packet_rate_hz": first_meta["config"]["radio"]["packet_rate_hz"],
                                        "detector.window_s": first_meta["config"]["detector"]["window_s"],
                                        "training.models": families})
            # Calibration uses precisely the same validation sessions as the original run.
            print(f"Recalibrating {folder.name}", flush=True)
            thresholds = calibrate(models, paths["validation"], protocol_cfg, folder)
            bundles[str(old_folder)] = {"models": models, "description": description, "thresholds": thresholds,
                                         "cfg": protocol_cfg, "folder": folder, "paths": paths}
        for name, condition in conditions.items():
            paths = [Path(p) for p in condition["cache_paths"]]
            specs = []
            for path in paths:
                _, _, meta = load_cache(path)
                specs.append({"scene_id": meta["scene_id"], "noise_seed": meta["noise_seed"], "mode": meta["mode"]})
            runner.evaluate_condition(name, condition["config"], bundles[condition["model_folder"]], condition["group"],
                                      condition["parameter"], condition["value"], families=["B2"] if name == "demo" else None,
                                      specs=specs, cached_paths=paths)
        compute_seconds = time.perf_counter() - start
        start = time.perf_counter()
        runner.finish()
        statistics_seconds = time.perf_counter() - start
        start = time.perf_counter()
        from csi_sim.plots import render_run
        render_run(destination)
        plot_seconds = time.perf_counter() - start
    corrections = {"source_run": str(source), "original_data_reused": True, "training_repeated": False,
                   "checkpoint_sha256": checkpoint_hashes,
                   "corrections": ["float64 probability-fusion weights and denominator",
                                   "retain -log(1-p) from model logits before sigmoid saturation",
                                   "weighted probability average through logsumexp of the complement"],
                   "threshold_source": "original validation sessions only",
                   "primary_metrics": "this corrected run", "original_run_role": "preserved numerical-diagnostic record"}
    write_json(destination / "numerical_correction.v1.json", corrections)
    resources = {"wall_seconds": old_resources["wall_seconds"] + monitor.elapsed_s,
                 "compute_seconds": old_resources["compute_seconds"] + compute_seconds,
                 "statistics_seconds": old_resources["statistics_seconds"] + statistics_seconds,
                 "plot_seconds": old_resources["plot_seconds"] + plot_seconds,
                 "peak_process_rss_bytes": max(old_resources["peak_process_rss_bytes"], monitor.peak_bytes),
                 "original_run_wall_seconds": old_resources["wall_seconds"], "reevaluation_wall_seconds": monitor.elapsed_s,
                 "timing_definition": "sum of original core execution and numerical-correction replay"}
    write_json(destination / "resources.json", resources)
    write_json(destination / "run_status.json", {"status": "completed", "conditions": list(runner.conditions),
                                               "figures_rendered": True, "score_domain": "negative_log_static_probability",
                                               "source_run": str(source)})
    print(f"Corrected results: {destination}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recompute(args.source, args.output)
