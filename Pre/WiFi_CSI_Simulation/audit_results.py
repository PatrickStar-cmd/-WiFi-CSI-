"""Check completed core outputs and protocol pairing without retraining models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from csi_sim.utils import read_csv, write_json


def metadata(path):
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["metadata"])


def audit(run_dir):
    root = Path(run_dir).resolve()
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    conditions = json.loads((root / "conditions.json").read_text(encoding="utf-8"))
    train, validation, test = (set(split[key]) for key in ("train", "validation", "test"))
    assert not (train & validation or train & test or validation & test)
    assert len(train | validation | test) == cfg["dataset"]["sessions"]
    all_caches = sorted(root.glob("models/*/data/*/*.h5")) + sorted(root.glob("conditions/*/data/*.h5"))
    samples, models = {}, {}
    for path in all_caches:
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["metadata"])
            valid = f["windows/valid"][...]
            features = f["windows/features"][...]
            times = f["windows/time"][...]
            assert features.shape == (*valid.shape, 16)
            assert np.isfinite(features[valid]).all()
            assert np.all(np.diff(times) > 0)
            assert len(f["truth/label"]) == len(times)
            if "amplitude" in f["windows"]:
                a = f["windows/amplitude"]
                assert a.shape[:3] == (len(times), 4, 56)
            if meta["split"] in {"train", "validation"}:
                assert meta["scene_id"] in split[meta["split"]]
    expected = {(scene, noise) for scene in test for noise in cfg["dataset"]["test_noise_seeds"]}
    for name in ("nominal", "fixed_budget_four"):
        paths = [Path(p) for p in conditions[name]["cache_paths"]]
        actual = {(m["scene_id"], m["noise_seed"]) for m in map(metadata, paths)}
        assert actual == expected and len(paths) == len(expected)
        samples[name] = {"mothers": len(test), "noise_repeats": len(cfg["dataset"]["test_noise_seeds"]), "sessions": len(paths)}
    pairing = 0
    for left in map(Path, conditions["nominal"]["cache_paths"]):
        right = root / "conditions/fixed_budget_four/data" / left.name
        lm, rm = metadata(left), metadata(right)
        assert lm["scene"] == rm["scene"]
        assert lm["scatterers"] == rm["scatterers"]
        assert lm["wall_coefficients_abs"] == rm["wall_coefficients_abs"]
        assert np.isclose(lm["body_scale"], rm["body_scale"])
        pairing += 1
    for folder in sorted((root / "models").iterdir()):
        cal = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
        desc = json.loads((folder / "models.json").read_text(encoding="utf-8"))
        for model, topologies in cal.items():
            for topology, record in topologies.items():
                assert record["validation_fpr"] <= cfg["detector"]["validation_fpr_limit"] + 1e-12
                assert np.isfinite(record["threshold"])
            if "best_single" in topologies:
                link = topologies["best_single"]["link"]
                assert link in ("AC", "AD", "BC", "BD")
                assert topologies["best_single"]["threshold"] == topologies[link]["threshold"]
        models[folder.name] = {"model_count": len(cal), "cnn_checkpoints": len(list(folder.glob("B3_seed*.pt"))),
                               "train_examples": desc["training"]["train_examples"], "validation_examples": desc["training"]["validation_examples"]}
    figures = [root / "figures" / name for name in ("scene_animation.gif", "scene_snapshot.png", "detection_timeline.png", "single_vs_four_links.png")]
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    probability_rows = 0
    for file in root.glob("conditions/*/predictions/*.csv"):
        for row in read_csv(file):
            if row.get("score_domain") == "negative_log_static_probability" and row.get("probability"):
                p, score, threshold = (float(row[key]) for key in ("probability", "score", "threshold"))
                assert 0 <= p <= 1 and score >= 0 and np.isfinite(threshold)
                probability_rows += 1
    result = {"status": "passed", "dataset_grouping": "disjoint", "caches_checked": len(all_caches),
              "protocol_scene_pairs": pairing, "samples": samples, "models": models,
              "validation_fpr_constraints": "satisfied", "finite_valid_features": True,
              "probability_range_rows_checked": probability_rows,
              "figure_files": {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in figures}}
    write_json(root / "audit.v1.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    audit(parser.parse_args().run_dir)
