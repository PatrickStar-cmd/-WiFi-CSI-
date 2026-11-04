"""Continuous torso/hand trajectories and independent ground-truth events."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .utils import generator, jsonable


def envelope(t: np.ndarray, duration: float, ramp: float) -> np.ndarray:
    ramp = min(ramp, duration / 3)
    edge = np.minimum(np.clip(t, 0, duration), np.clip(duration - t, 0, duration))
    return np.where(edge < ramp, 0.5 * (1 - np.cos(np.pi * edge / ramp)), 1.0)


def distance_fraction(t: np.ndarray, duration: float, ramp: float) -> np.ndarray:
    """Integral of a unit-speed cosine-ramped motion, normalized to [0, 1]."""
    ramp = min(ramp, duration / 3)
    t = np.clip(t, 0, duration)
    start = 0.5 * (t - ramp / np.pi * np.sin(np.pi * t / ramp))
    remaining = duration - t
    tail = duration - ramp - 0.5 * (remaining - ramp / np.pi * np.sin(np.pi * remaining / ramp))
    integrated = np.where(t < ramp, start, np.where(t > duration - ramp, tail, t - ramp / 2))
    return integrated / (duration - ramp)


def make_legs(origin, heading, start, end, speed, cfg) -> list[dict]:
    """Straight legs connected at zero velocity; reflected headings keep the body inside."""
    box = np.asarray(cfg["room"]["motion_box_m"], dtype=float)
    point = np.asarray(origin, dtype=float).copy()
    direction = np.array([np.cos(heading), np.sin(heading)])
    ramp = cfg["motion"]["ramp_s"]
    clock = start
    result = []
    while end - clock > 1e-8:
        remaining = end - clock
        wall_distances = np.full(2, np.inf)
        for axis in range(2):
            if abs(direction[axis]) > 1e-10:
                boundary = box[axis, 1 if direction[axis] > 0 else 0]
                wall_distances[axis] = max(0, (boundary - point[axis]) / direction[axis])
        hit_axis = int(np.argmin(wall_distances))
        distance_to_wall = wall_distances[hit_axis]
        if distance_to_wall < 1e-7:
            direction[hit_axis] *= -1
            continue
        max_distance = speed * (remaining - min(ramp, remaining / 3))
        distance = min(distance_to_wall, max_distance)
        duration = remaining if distance >= max_distance - 1e-9 else distance / speed + ramp
        # Very short legs use a shorter symmetric ramp.
        if duration < 3 * ramp and distance < max_distance - 1e-9:
            duration = 1.5 * distance / speed
        target = point + distance * direction
        result.append({"start": clock, "end": clock + duration, "p0": point.copy(), "p1": target.copy()})
        point, clock = target, clock + duration
        if distance >= distance_to_wall - 1e-8:
            direction[hit_axis] *= -1
    return result


@dataclass
class Scene:
    cfg: dict
    scene_id: int
    segments: list[dict]
    duration: float
    present: bool = True

    def positions(self, times) -> np.ndarray:
        times = np.atleast_1d(times).astype(float)
        m = self.cfg["motion"]
        initial = np.asarray(self.segments[0]["p0"], dtype=float)
        points = np.repeat(initial[None, None, :], len(times) * 3, axis=0).reshape(len(times), 3, 2)
        points[:, 1, 1] += m["hand_offset_m"]
        points[:, 2, 1] -= m["hand_offset_m"]
        for segment in self.segments:
            mask = (times >= segment["start"]) & (times < segment["end"])
            if segment is self.segments[-1]:
                mask |= times >= segment["end"]
            if not mask.any():
                continue
            local = times[mask]
            body = np.repeat(np.asarray(segment["p0"])[None, :], len(local), axis=0)
            swing = np.zeros_like(body)
            if segment["kind"] == "walk":
                for leg in segment["legs"]:
                    lm = (local >= leg["start"]) & (local < leg["end"])
                    if leg is segment["legs"][-1]:
                        lm |= local >= leg["end"]
                    dt, length = local[lm] - leg["start"], leg["end"] - leg["start"]
                    vector = np.asarray(leg["p1"]) - leg["p0"]
                    unit = vector / max(np.linalg.norm(vector), 1e-12)
                    body[lm] = leg["p0"] + distance_fraction(dt, length, m["ramp_s"])[:, None] * vector
                    gait = m["gait_amplitude_m"] * envelope(dt, length, m["ramp_s"]) * np.sin(2 * np.pi * m["gait_hz"] * dt)
                    swing[lm] = gait[:, None] * unit
            out = np.repeat(body[:, None, :], 3, axis=1)
            out[:, 1, 1] += m["hand_offset_m"]
            out[:, 2, 1] -= m["hand_offset_m"]
            out[:, 1] += swing
            out[:, 2] -= swing
            if segment["kind"] == "wave":
                dt = local - segment["start"]
                wave = m["wave_amplitude_m"] * envelope(dt, segment["end"] - segment["start"], m["ramp_s"]) * np.sin(2 * np.pi * m["wave_hz"] * dt)
                out[:, 2, 0] += wave
            points[mask] = out
        return points

    def labels(self, times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.asarray(times)
        y = np.zeros(len(times), dtype=np.int8)
        kinds = np.full(len(times), "still" if self.present else "empty", dtype="U8")
        event_ids = np.full(len(times), -1, dtype=int)
        for index, segment in enumerate(self.segments):
            mask = (times >= segment["start"]) & (times < segment["end"])
            y[mask] = segment["kind"] in {"walk", "wave"}
            kinds[mask] = segment["kind"]
            event_ids[mask] = index if segment["kind"] in {"walk", "wave"} else -1
        return y, kinds, event_ids

    def fractions(self, ends, window_s) -> np.ndarray:
        result = np.zeros(len(ends), dtype=float)
        for event in self.events:
            result += np.maximum(0, np.minimum(ends, event["end"]) - np.maximum(ends - window_s, event["start"]))
        return np.clip(result / window_s, 0, 1)

    @property
    def events(self) -> list[dict]:
        return [{"start": s["start"], "end": s["end"], "kind": s["kind"], "event_id": i}
                for i, s in enumerate(self.segments) if s["kind"] in {"walk", "wave"}]

    def as_dict(self):
        return jsonable({"scene_id": self.scene_id, "duration": self.duration, "present": self.present,
                         "segments": self.segments, "events": self.events})


def make_scene(cfg: dict, scene_id: int, mode="random", origin=None, heading=None) -> Scene:
    rng = generator(cfg, scene_id, "trajectory")
    cal = cfg["dataset"]["calibration_s"]
    duration = cfg["dataset"]["duration_s"]
    box = np.asarray(cfg["room"]["motion_box_m"])
    point = np.array([rng.uniform(*row) for row in box]) if origin is None else np.asarray(origin, dtype=float)
    if mode == "demo":
        point, duration = np.array([2.0, 2.5]), cal + 25.0
        schedule = [("still", cal + 5), ("walk", cal + 10), ("still", cal + 15), ("wave", cal + 20), ("still", duration)]
    elif mode in {"static", "empty"}:
        duration = cal + cfg["dataset"]["static_test_s"]
        schedule = [("empty" if mode == "empty" else "still", duration)]
    elif mode == "coverage":
        move_duration = cfg["experiments"]["coverage_distance_m"] / cfg["motion"]["walk_speed_mps"] + cfg["motion"]["ramp_s"]
        duration = cal + 3 + move_duration + 3
        schedule = [("still", cal + 3), ("walk", cal + 3 + move_duration), ("still", duration)]
    else:
        schedule = [("still", cal)]
        current = cal
        moving = bool(rng.integers(0, 2))
        while current < duration - 1e-9:
            current = min(duration, current + rng.uniform(*cfg["dataset"]["segment_duration_s"]))
            kind = str(rng.choice(["walk", "wave"])) if moving else "still"
            schedule.append((kind, current))
            moving = not moving
    start, segments = 0.0, []
    for kind, end in schedule:
        segment = {"kind": kind, "start": start, "end": end, "p0": point.copy(), "legs": []}
        if kind == "walk":
            angle = 0.0 if mode == "demo" else (float(rng.uniform(-np.pi, np.pi)) if heading is None else heading)
            speed = 0.4 if mode == "demo" else cfg["motion"]["walk_speed_mps"]
            segment["legs"] = make_legs(point, angle, start, end, speed, cfg)
            point = np.asarray(segment["legs"][-1]["p1"])
        segments.append(segment)
        start = end
    return Scene(cfg, scene_id, segments, duration, mode != "empty")
