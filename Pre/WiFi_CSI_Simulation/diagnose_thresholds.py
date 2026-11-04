"""Validation-only operating-point diagnosis; trained weights stay fixed."""
import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

from csi_sim.config import changed
from csi_sim.dataset import load_cache
from csi_sim.fusion import confirm_states, fuse, select_threshold
from csi_sim.models import predict
from csi_sim.utils import write_csv, write_json


def diagnose(run_dir, output):
    start = time.perf_counter()
    run_dir, output = Path(run_dir), Path(output)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    folder = next(p for p in (run_dir / "models").iterdir() if p.name.startswith("rate100_"))
    model = {"kind": "B2", "model": joblib.load(folder / "B2.joblib")}
    sequences = {key: [] for key in ("AC", "AD", "BC", "BD", "four_equal", "four_quality")}
    for path in sorted((folder / "data/validation").glob("*.h5")):
        data, truth, _ = load_cache(path)
        scores = predict(model, path, data, cfg)
        for topology in sequences:
            fused, _ = fuse(scores, data["valid"], data["quality"], topology, cfg["detector"]["weight_ratio_limit"], "negative_log_static_probability")
            sequences[topology].append({"score": fused, "truth": truth["label"]})
    original = {key: select_threshold(value, cfg) for key, value in sequences.items()}
    best = max(("AC", "AD", "BC", "BD"), key=lambda key: (original[key]["validation_f1"], -original[key]["validation_fpr"]))
    rows = []
    for limit in (0.01, 0.03, 0.05, 0.10):
        local = changed(cfg, {"detector.validation_fpr_limit": limit})
        for topology in (best, "four_equal", "four_quality"):
            record = select_threshold(sequences[topology], local)
            tp = fp = fn = tn = 0
            for sequence in sequences[topology]:
                state = confirm_states(sequence["score"], record["threshold"], cfg["detector"]["confirmation_count"])
                y = sequence["truth"]
                tp += int(np.sum((state == 1) & (y == 1)))
                fp += int(np.sum((state == 1) & (y == 0)))
                fn += int(np.sum((state == 0) & (y == 1)))
                tn += int(np.sum((state == 0) & (y == 0)))
            rows.append({"split": "validation_only", "fpr_limit": limit, "topology": topology, **record,
                         "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else None,
                         "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    write_csv(output.parent / f"{output.name}.csv", rows)
    result = {"status": "validation_diagnostic", "primary_fpr_limit": 0.01, "fixed_single_link": best,
              "training_repeated": False, "test_data_used": False, "seconds": time.perf_counter() - start, "rows": rows}
    write_json(output.parent / f"{output.name}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run_dir, args.output)
