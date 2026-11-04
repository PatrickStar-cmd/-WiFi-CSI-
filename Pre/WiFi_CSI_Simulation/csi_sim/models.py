"""Fit on training mothers, select on validation mothers, freeze for tests."""
from __future__ import annotations

from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import collect_examples, load_cache
from .fusion import TOPOLOGIES, fuse, select_threshold
from .utils import write_json


def fit_models(train_paths, val_paths, cfg, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    models, descriptions = {}, {}
    train, validation = collect_examples(train_paths), collect_examples(val_paths)
    start = time.perf_counter()
    for name in cfg["training"]["models"]:
        if name in {"B0", "B0q", "B1", "B1a"}:
            models[name] = {"kind": name}
            descriptions[name] = {"kind": name, "parameter_count": 1}
        elif name == "B2":
            candidates = []
            for c in cfg["training"]["lr_c"]:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=2000))
                model.fit(train[0], train[1])
                loss = float(log_loss(validation[1], model.predict_proba(validation[0])[:, 1], labels=[0, 1]))
                candidates.append((loss, c, model))
            loss, c, model = min(candidates, key=lambda x: (x[0], x[1]))
            joblib.dump(model, folder / "B2.joblib")
            models[name] = {"kind": name, "model": model}
            descriptions[name] = {"kind": name, "C": c, "validation_log_loss": loss, "parameter_count": 17,
                                  "candidate_log_losses": [{"C": candidate[1], "loss": candidate[0]} for candidate in candidates]}
        elif name == "B3":
            from .cnn import train_cnn
            for seed in cfg["training"]["cnn_seeds"]:
                tag = f"B3_seed{seed}"
                artifact = train_cnn(train, validation, train_paths, cfg, seed, folder / f"{tag}.pt")
                models[tag] = {"kind": name, "artifact": artifact}
                descriptions[tag] = {"kind": name, "seed": seed, "validation_log_loss": artifact["validation_loss"], "parameter_count": 7121}
    descriptions["training"] = {"seconds": time.perf_counter() - start, "train_examples": len(train[1]), "validation_examples": len(validation[1])}
    write_json(folder / "models.json", descriptions)
    return models, descriptions


def predict(model, cache_path, data, cfg):
    kind, valid = model["kind"], data["valid"]
    if kind in {"B0", "B0q", "B1", "B1a"}:
        return data["baselines"][:, :, {"B0": 0, "B0q": 1, "B1": 2, "B1a": 3}[kind]].astype(float)
    if kind == "B2":
        score = np.full(valid.shape, np.nan)
        if valid.any():
            score[valid] = np.logaddexp(0.0, model["model"].decision_function(data["features"][valid]))
        return score
    if kind == "B3":
        from .cnn import predict_cnn
        return predict_cnn(model["artifact"], cache_path, valid, cfg["training"]["batch_size"])
    raise ValueError(kind)


def calibrate(models, val_paths, cfg, folder):
    calibration = {}
    for name, model in models.items():
        domain = "negative_log_static_probability" if model["kind"] in {"B2", "B3"} else "linear"
        topologies = list(TOPOLOGIES) if model["kind"] in {"B2", "B3"} else ["AC"]
        sequences = {topology: [] for topology in topologies}
        for path in val_paths:
            data, truth, _ = load_cache(path)
            score = predict(model, path, data, cfg)
            for topology in topologies:
                fused, _ = fuse(score, data["valid"], data["quality"], topology, cfg["detector"]["weight_ratio_limit"], domain)
                sequences[topology].append({"score": fused, "truth": truth["label"]})
        calibration[name] = {topology: {**select_threshold(seq, cfg), "score_domain": domain} for topology, seq in sequences.items()}
        if model["kind"] in {"B2", "B3"}:
            best = max(("AC", "AD", "BC", "BD"), key=lambda link: (calibration[name][link]["validation_f1"], -calibration[name][link]["validation_fpr"]))
            calibration[name]["best_single"] = {**calibration[name][best], "link": best}
    write_json(Path(folder) / "calibration.json", calibration)
    return calibration
