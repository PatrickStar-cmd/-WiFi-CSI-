"""Grouped sessions, causal windows, and HDF5 observation/truth separation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from .channel import Channel, generate_observations, tones
from .config import LINKS, changed, fingerprint
from .features import compute_features
from .scene import make_scene
from .utils import generator, jsonable, seed_for


def split_ids(cfg):
    d = cfg["dataset"]
    ids = generator(cfg, 0, "split").permutation(d["sessions"])
    a, b = d["train_sessions"], d["train_sessions"] + d["validation_sessions"]
    return {"train": ids[:a].tolist(), "validation": ids[a:b].tolist(), "test": ids[b:].tolist()}


def session_config(cfg, scene_id, split):
    if split not in {"train", "validation"}:
        return cfg
    rng = generator(cfg, scene_id, "training_conditions")
    d = cfg["dataset"]
    return changed(cfg, {"motion.walk_speed_mps": float(rng.uniform(*d["train_speed_range"])),
                         "observation.snr_db": float(rng.uniform(*d["train_snr_range"])),
                         "observation.body_ratio_db": float(rng.uniform(*d["train_body_ratio_range"]))})


def extract_windows(observations, cfg, duration, store_amplitude=True):
    """Access CSI, receive times and gain metadata only; labels are attached later."""
    clock_start = time.perf_counter()
    d, det, rate = cfg["dataset"], cfg["detector"], cfg["radio"]["packet_rate_hz"]
    times = np.arange(d["calibration_s"], duration - 1e-9, det["step_s"])
    n, nt, nk = len(times), round(det["window_s"] * rate), len(tones(cfg))
    features = np.full((n, 4, 16), np.nan, dtype=np.float32)
    baselines = np.full((n, 4, 4), np.nan, dtype=np.float32)
    amplitude = np.zeros((n, 4, nk, nt), dtype=np.float32) if store_amplitude else None
    valid = np.zeros((n, 4), dtype=bool)
    quality = np.zeros((n, 4), dtype=np.float32)
    valid_ratio = np.zeros((n, 4), dtype=np.float32)
    calibration_rms, calibration_noise = [], []
    eps = det["epsilon"]
    for li, link in enumerate(LINKS):
        o = observations[link]
        a = np.abs(o["csi"]).astype(float)
        if cfg["observation"]["gain_compensation"]:
            a /= 10 ** (o["gain_db"][:, None] / 20)
        finite = o["valid"] & np.isfinite(a).all(axis=1)
        cal = finite & (o["timestamp"] >= 0) & (o["timestamp"] < d["calibration_s"]) & (o["arrival_time"] < d["calibration_s"])
        if cal.sum() < 2:
            raise ValueError(f"{link}: calibration requires at least two valid packets.")
        scale = float(np.sqrt(np.mean(a[cal] ** 2)))
        if scale <= 0:
            raise ValueError(f"{link}: zero calibration amplitude.")
        a /= scale
        noise = float(max(np.median(a[cal].var(axis=0)), eps))
        calibration_rms.append(scale)
        calibration_noise.append(noise)
        stamp = o["timestamp"]
        for wi, end in enumerate(times):
            start = end - det["window_s"]
            left = np.searchsorted(stamp, start, side="right")
            right = np.searchsorted(stamp, end + 1e-10, side="right")
            indices = np.arange(left, right)
            indices = indices[finite[indices] & (o["arrival_time"][indices] <= end + 1e-10)]
            ratio = min(len(indices) / nt, 1.0)
            valid_ratio[wi, li] = ratio
            if ratio + 1e-9 < det["min_valid_ratio"] or len(indices) < 3:
                continue
            t = stamp[indices]
            if np.max(np.diff(np.r_[start, t, end])) > det["max_gap_s"] + 1e-9:
                continue
            adjacent = (np.diff(o["packet_id"][indices]) == 1) & (np.diff(t) > 1e-8)
            if adjacent.sum() < 2:
                continue
            raw = a[indices]
            grid = end - np.arange(nt - 1, -1, -1) / rate
            # Interpolation uses samples already available at the decision time.
            filled = np.column_stack([np.interp(grid, t, raw[:, k]) for k in range(nk)])
            f, b = compute_features(raw, t, o["packet_id"][indices], filled, rate, eps, 20 * np.log10(scale))
            features[wi, li], baselines[wi, li] = f, b
            if store_amplitude:
                amplitude[wi, li] = filled.T
            valid[wi, li] = True
            quality[wi, li] = ratio / noise
    result = {"time": times, "features": features, "baselines": baselines,
            "valid": valid, "quality": quality, "valid_ratio": valid_ratio,
            "calibration_rms": np.asarray(calibration_rms), "calibration_noise": np.asarray(calibration_noise),
            "feature_seconds": time.perf_counter() - clock_start}
    if store_amplitude:
        result["amplitude"] = amplitude
    return result


def create_cache(path, cfg, scene_id, split, noise_seed=0, mode="random", save_raw=True, save_amplitude=True, origin=None, heading=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    local_cfg = session_config(cfg, scene_id, split)
    scene = make_scene(local_cfg, scene_id, mode, origin, heading)
    channel = Channel(local_cfg, scene)
    observations = generate_observations(local_cfg, scene, channel, noise_seed)
    windows = extract_windows(observations, local_cfg, scene.duration, save_amplitude)
    y, kind, event_id = scene.labels(windows["time"])
    fraction = scene.fractions(windows["time"], local_cfg["detector"]["window_s"])
    session_id = f"s{scene_id:06d}_n{noise_seed}"
    metadata = {"scene_id": scene_id, "session_id": session_id, "split": split, "noise_seed": noise_seed,
                "mode": mode, "config": local_cfg, "config_hash": fingerprint(local_cfg), "scene": scene.as_dict(),
                "nodes": channel.nodes, "scatterers": channel.scatterers,
                "wall_coefficients_abs": abs(channel.wall_coefficients),
                "background_power": channel.reference_background_power, "body_scale": channel.body_scale,
                "noise_variance": channel.reference_background_power * 10 ** (-local_cfg["observation"]["snr_db"] / 10),
                "seeds": {key: seed_for(cfg["seed"], scene_id, key) for key in ("background", "trajectory", "training_conditions")},
                "packet_counts": {link: len(o["timestamp"]) for link, o in observations.items()},
                "packet_valid_ratio": {link: float(o["valid"].mean()) for link, o in observations.items()},
                "feature_seconds": windows.pop("feature_seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps(jsonable(metadata), ensure_ascii=False)
        f.create_dataset("subcarrier_index", data=channel.k)
        group = f.create_group("windows")
        for key, value in windows.items():
            if key == "amplitude" and not save_amplitude:
                continue
            chunks = (1, 1, value.shape[2], value.shape[3]) if key == "amplitude" else None
            group.create_dataset(key, data=value, compression="gzip", compression_opts=1, chunks=chunks)
        truth = f.create_group("truth")
        for key, value in {"label": y, "motion_fraction": fraction, "event_id": event_id,
                           "kind": kind.astype("S8"), "torso_position": scene.positions(windows["time"] - local_cfg["detector"]["window_s"] / 2)[:, 0]}.items():
            truth.create_dataset(key, data=value)
        if save_raw:
            raw = f.create_group("observations")
            for link, o in observations.items():
                lg = raw.create_group(link)
                for key, value in o.items():
                    lg.create_dataset(key, data=value, compression="gzip", compression_opts=1)
            truth_time = np.arange(0, scene.duration, 0.02)
            truth.create_dataset("trajectory_time", data=truth_time)
            truth.create_dataset("trajectory_positions", data=scene.positions(truth_time))
            truth.create_dataset("trajectory_label", data=scene.labels(truth_time)[0])
    return path


def load_cache(path, with_amplitude=False):
    with h5py.File(path, "r") as f:
        data = {key: f["windows"][key][...] for key in f["windows"] if with_amplitude or key != "amplitude"}
        truth = {key: f["truth"][key][...] for key in ("label", "motion_fraction", "event_id", "kind", "torso_position")}
        truth["kind"] = truth["kind"].astype(str)
        meta = json.loads(f.attrs["metadata"])
    return data, truth, meta


def collect_examples(paths):
    features, labels, refs = [], [], []
    for path in paths:
        data, truth, _ = load_cache(path)
        fraction = truth["motion_fraction"]
        pure = (fraction <= 0.2 + 1e-9) | (fraction >= 0.8 - 1e-9)
        rows, links = np.where(data["valid"] & pure[:, None])
        features.append(data["features"][rows, links])
        labels.append((fraction[rows] >= 0.8 - 1e-9).astype(np.int64))
        refs.extend((str(path), int(row), int(link)) for row, link in zip(rows, links))
    x, y = np.concatenate(features), np.concatenate(labels)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training/validation examples require both motion and static labels.")
    return x, y, refs
