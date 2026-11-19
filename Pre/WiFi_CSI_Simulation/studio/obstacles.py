"""Finite, oriented obstacles for scene validation and geometric propagation."""
from dataclasses import dataclass

import numpy as np


def horizontal_axes(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Prism:
    record: dict
    center: np.ndarray
    half_size: np.ndarray
    axes: np.ndarray
    kind: str

    def local(self, point):
        return (np.asarray(point, dtype=float) - self.center) @ self.axes

    def contains(self, point):
        return np.all(np.abs(self.local(point)) <= self.half_size + 1e-9, axis=-1)

    def crosses(self, start, end):
        """Test whether each finite segment has an interval inside the prism."""
        a, b = self.local(start), self.local(end)
        delta = b - a
        parallel = np.abs(delta) < 1e-9
        safe_delta = np.where(parallel, 1.0, delta)
        low = (-self.half_size - a) / safe_delta
        high = (self.half_size - a) / safe_delta
        entry = np.max(np.where(parallel, -np.inf, np.minimum(low, high)), axis=-1)
        exit = np.min(np.where(parallel, np.inf, np.maximum(low, high)), axis=-1)
        outside_parallel = np.any(parallel & (np.abs(a) >= self.half_size - 1e-9), axis=-1)
        return ~outside_parallel & (np.minimum(exit, 1.0) > np.maximum(entry, 0.0) + 1e-9)

    def reflection_point(self, start, end):
        """Find a mirror reflection on either of the wall's two broad faces."""
        a, b = self.local(start), self.local(end)
        side = 1 if a[1] >= 0 else -1
        face = side * self.half_size[1]
        if side * (a[1] - face) <= 1e-9 or side * (b[1] - face) <= 1e-9:
            return None
        mirror = a.copy()
        mirror[1] = 2 * face - a[1]
        fraction = (face - mirror[1]) / (b[1] - mirror[1])
        hit = mirror + fraction * (b - mirror)
        if abs(hit[0]) > self.half_size[0] or abs(hit[2]) > self.half_size[2]:
            return None
        return self.center + hit @ self.axes.T


def wall_prism(wall):
    a, b = np.asarray([wall['start'], wall['end']], dtype=float)
    delta = b - a
    center = np.r_[(a + b) / 2, wall['height'] / 2]
    half_size = np.array([np.linalg.norm(delta), wall['thickness'], wall['height']]) / 2
    return Prism(wall, center, half_size, horizontal_axes(np.arctan2(delta[1], delta[0])), 'wall')


def box_prism(furniture):
    half_size = np.asarray(furniture['size'], dtype=float) / 2
    center = np.asarray(furniture['position'], dtype=float) + [0, 0, half_size[2]]
    angle = np.deg2rad(furniture.get('heading', 0))
    return Prism(furniture, center, half_size, horizontal_axes(angle), 'box')
