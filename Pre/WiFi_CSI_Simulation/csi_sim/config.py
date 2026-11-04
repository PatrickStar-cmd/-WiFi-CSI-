"""Configuration handling uses only the Python standard library."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

LINKS = ("AC", "AD", "BC", "BD")


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    r, d, w = cfg["radio"], cfg["dataset"], cfg["detector"]
    assert r["nfft"] == 64 and r["cp"] == 16, "The reference waveform uses FFT64 / CP16."
    assert r["tone_max"] == 28 and r["tone_min"] == 1
    assert r["packet_rate_hz"] > 0 and r["sample_rate_hz"] > 0
    assert d["train_sessions"] > 0 and d["validation_sessions"] > 0
    assert d["sessions"] > d["train_sessions"] + d["validation_sessions"]
    assert d["duration_s"] > d["calibration_s"] >= w["window_s"]
    assert 0 < w["min_valid_ratio"] <= 1 and w["confirmation_count"] >= 1
    assert 0 <= cfg["observation"]["loss_rate"] < 1
    assert cfg["observation"]["gain_mode"] in {"none", "drift", "step"}
    assert set(cfg["training"]["models"]) <= {"B0", "B0q", "B1", "B1a", "B2", "B3"}
    assert "B2" in cfg["training"]["models"], "B2 is the primary topology comparison model."
    for key in ("room", "motion", "experiments", "plots"):
        assert key in cfg


def changed(cfg: dict, overrides: dict) -> dict:
    result = copy.deepcopy(cfg)
    for path, value in overrides.items():
        keys = path.split(".")
        node = result
        for key in keys[:-1]:
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(path)
        node[keys[-1]] = value
    validate_config(result)
    return result


def fingerprint(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def experiment_plan(cfg: dict) -> dict:
    """Describe planned work; this function creates no samples or outputs."""
    d = cfg["dataset"]
    return {
        "status": "ready_to_run",
        "split": {"train": d["train_sessions"], "validation": d["validation_sessions"],
                  "test": d["sessions"] - d["train_sessions"] - d["validation_sessions"]},
        "test_noise_repeats": len(d["test_noise_seeds"]),
        "models": cfg["training"]["models"],
        "cnn_initializations": cfg["training"]["cnn_seeds"],
        "stages": {"E0": "geometry, Doppler and OFDM estimation",
                   "E1": "AC detector comparison and static false alarms",
                   "E2": "single/dual/four links and two transmit budgets",
                   "E3": "SNR, speed, body return and waving sweeps",
                   "E4": "packet rate, window and loss sweeps",
                   "E5": "gain, timing, outages, fan and receiver displacement",
                   "demo": "25-second motion sequence and synchronized visualization",
                   "coverage": "0.5-m spatial grid, eight movement directions",
                   "region": "optional left/centre/right classifier"},
        "output": ["HDF5", "JSON", "CSV", "PNG", "SVG", "GIF"],
    }
