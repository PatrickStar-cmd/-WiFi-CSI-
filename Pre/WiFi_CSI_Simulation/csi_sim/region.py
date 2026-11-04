"""Optional, independently trained left/centre/right classifier."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_cache
from .utils import write_csv, write_json


def examples(paths, include_transition=False):
    x, y, identity = [], [], []
    for path in paths:
        data, truth, meta = load_cache(path)
        position = truth["torso_position"][:, 0]
        boundary = (abs(position - 2) < 0.2) | (abs(position - 4) < 0.2)
        valid = data["valid"].all(axis=1) & (truth["motion_fraction"] >= 0.8 - 1e-9)
        valid &= boundary if include_transition else ~boundary
        indices = np.flatnonzero(valid)
        x.extend(data["features"][indices].reshape(len(indices), 64))
        y.extend(np.digitize(position[indices], [2.0, 4.0]))
        identity.extend({"scene_id": meta["scene_id"], "session_id": meta["session_id"], "time": data["time"][i]} for i in indices)
    return np.asarray(x).reshape(-1, 64), np.asarray(y, dtype=int), identity


def run_region(bundle, test_paths, cfg, output):
    output = Path(output)
    output.mkdir(parents=True)
    train_x, train_y, _ = examples(bundle["paths"]["train"])
    val_x, val_y, _ = examples(bundle["paths"]["validation"])
    if set(np.unique(train_y)) != {0, 1, 2}:
        raise ValueError("Region training requires examples from all three regions.")
    candidates = []
    for c in cfg["training"]["lr_c"]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(train_x, train_y)
        candidates.append((log_loss(val_y, model.predict_proba(val_x), labels=[0, 1, 2]), c, model))
    loss, c, model = min(candidates, key=lambda item: (item[0], item[1]))
    joblib.dump(model, output / "region.joblib")
    metrics = {"C": c, "validation_log_loss": loss, "region_labels": ["left", "centre", "right"]}
    for name, transition in (("main", False), ("boundary", True)):
        x, y, rows = examples(test_paths, transition)
        if not len(y):
            metrics[name] = {"samples": 0}
            continue
        predictions = model.predict(x)
        metrics[name] = {"samples": len(y), "macro_f1": f1_score(y, predictions, average="macro", labels=[0, 1, 2], zero_division=0),
                         "confusion": confusion_matrix(y, predictions, labels=[0, 1, 2]).tolist()}
        write_csv(output / f"{name}_predictions.csv", [{**row, "truth": int(t), "prediction": int(p)} for row, t, p in zip(rows, y, predictions)])
    write_json(output / "metrics.json", metrics)
