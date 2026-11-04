from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np


def seed_for(base: int, scene_id: int, stream: str, repeat: int = 0) -> int:
    key = f"{base}:{scene_id}:{stream}:{repeat}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def generator(cfg: dict, scene_id: int, stream: str, repeat: int = 0):
    return np.random.default_rng(seed_for(cfg["seed"], scene_id, stream, repeat))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: jsonable(v) for k, v in row.items()} for row in rows)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def environment() -> dict:
    names = ["numpy", "scipy", "scikit-learn", "matplotlib", "h5py", "torch", "Pillow", "psutil"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


class ResourceMonitor:
    """Sample this process's resident memory; no child processes are launched."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.start = time.perf_counter()
        self.peak_bytes = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self):
        while not self.stop.wait(0.1):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        self.elapsed_s = time.perf_counter() - self.start
