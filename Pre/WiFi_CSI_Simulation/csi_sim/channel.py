"""Shared geometric multipath and per-receiver observation impairments."""
from __future__ import annotations

import numpy as np

from .config import LINKS
from .utils import generator

C0 = 299_792_458.0


def tones(cfg):
    k = np.arange(cfg["radio"]["tone_min"], cfg["radio"]["tone_max"] + 1)
    return np.r_[-k[::-1], k]


class Channel:
    def __init__(self, cfg, scene):
        self.cfg, self.scene = cfg, scene
        self.k = tones(cfg)
        r, room = cfg["radio"], cfg["room"]
        self.frequencies = r["fc_hz"] + self.k * r["sample_rate_hz"] / r["nfft"]
        self.nominal_nodes = {k: np.asarray(v, dtype=float) for k, v in room["nodes"].items()}
        self.nodes = {k: v.copy() for k, v in self.nominal_nodes.items()}
        self.nodes["C"][0] += cfg["observation"]["node_c_shift_m"]
        rng = generator(cfg, scene.scene_id, "background")
        self.wall_coefficients = -rng.uniform(*room["wall_coefficient_range"], 4).astype(complex)
        low, high = np.array([0.6, 0.6]), np.asarray(room["size_m"]) - 0.6
        self.scatterers = rng.uniform(low, high, (room["static_scatterers"], 2))
        self.scatter_coefficients = rng.uniform(*room["scatter_coefficient_range"], len(self.scatterers)) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(self.scatterers)))
        self.static = [self._static_paths(link, self.nodes) for link in LINKS]
        nominal_static = [self._static_paths(link, self.nominal_nodes) for link in LINKS]
        nominal_bg = np.array([self._sum(d, a) for d, a in nominal_static])
        self.background = np.array([self._sum(d, a) for d, a in self.static])
        self.reference_background_power = np.mean(abs(nominal_bg) ** 2, axis=1)
        center = np.asarray(room["size_m"]) / 2
        body = np.repeat(center[None, :], 3, axis=0)
        body[1, 1] += cfg["motion"]["hand_offset_m"]
        body[2, 1] -= cfg["motion"]["hand_offset_m"]
        raw_power = np.mean([np.sum(abs(self._body_paths(link, body[None], self.nominal_nodes)[1]) ** 2) for link in LINKS])
        self.body_scale = np.sqrt(np.mean(self.reference_background_power) * 10 ** (cfg["observation"]["body_ratio_db"] / 10) / raw_power)
        fan = np.asarray(cfg["observation"]["fan_position_m"])[None, None, :]
        fan_power = np.mean([np.sum(abs(self._point_paths(link, fan, np.ones(1), self.nominal_nodes)[1]) ** 2) for link in LINKS])
        self.fan_scale = np.sqrt(np.mean(self.reference_background_power) * 10 ** (cfg["observation"]["fan_ratio_db"] / 10) / fan_power)

    def _sum(self, distance, amplitude):
        return np.sum(amplitude[..., None] * np.exp(-2j * np.pi * distance[..., None] * self.frequencies / C0), axis=-2)

    def _static_paths(self, link, nodes):
        tx, rx = nodes[link[0]], nodes[link[1]]
        width, height = self.cfg["room"]["size_m"]
        images = np.array([[-tx[0], tx[1]], [2 * width - tx[0], tx[1]], [tx[0], -tx[1]], [tx[0], 2 * height - tx[1]]])
        direct = np.linalg.norm(tx - rx)
        walls = np.linalg.norm(images - rx, axis=1)
        ds1 = np.linalg.norm(self.scatterers - tx, axis=1)
        ds2 = np.linalg.norm(self.scatterers - rx, axis=1)
        return (np.r_[direct, walls, ds1 + ds2],
                np.r_[1 / direct, self.wall_coefficients / walls, self.scatter_coefficients / (ds1 * ds2)])

    @staticmethod
    def _point_paths(link, points, coefficients, nodes):
        d1 = np.linalg.norm(points - nodes[link[0]], axis=-1)
        d2 = np.linalg.norm(points - nodes[link[1]], axis=-1)
        return d1 + d2, coefficients / np.maximum(d1 * d2, 1e-8)

    def _body_paths(self, link, points, nodes):
        return self._point_paths(link, points, np.asarray(self.cfg["motion"]["body_amplitudes"]), nodes)

    def body_distances(self, link, times):
        return self._body_paths(link, self.scene.positions(times), self.nodes)[0]

    def response(self, link, times):
        times = np.atleast_1d(times)
        result = np.repeat(self.background[LINKS.index(link)][None, :], len(times), axis=0)
        if self.scene.present:
            distance, amplitude = self._body_paths(link, self.scene.positions(times), self.nodes)
            result += self._sum(distance, self.body_scale * amplitude)
        if self.cfg["observation"]["fan"]:
            obs = self.cfg["observation"]
            points = np.repeat(np.asarray(obs["fan_position_m"])[None, None, :], len(times), axis=0)
            points[:, 0, 0] += obs["fan_amplitude_m"] * np.sin(2 * np.pi * obs["fan_hz"] * times)
            d, a = self._point_paths(link, points, np.ones(1), self.nodes)
            result += self._sum(d, self.fan_scale * a)
        return result

    def paths_at(self, link, time):
        """Physical distances and coefficients, before the carrier propagation phase."""
        d, a = self.static[LINKS.index(link)]
        if self.scene.present:
            bd, ba = self._body_paths(link, self.scene.positions([time]), self.nodes)
            d, a = np.r_[d, bd[0]], np.r_[a, self.body_scale * ba[0]]
        return d, a


def generate_observations(cfg, scene, channel, noise_seed=0):
    rate = cfg["radio"]["packet_rate_hz"]
    obs = cfg["observation"]
    base_times = np.arange(0, scene.duration, 1 / rate)
    observations = {}
    # Receiver gain and timestamp curves are shared by its A/B links.
    rx_grid = np.arange(0, scene.duration + 1 / rate, 1 / (2 * rate))
    rx_curves = {}
    for ri, receiver in enumerate("CD"):
        rng = generator(cfg, scene.scene_id, f"receiver_{receiver}", noise_seed)
        gain = np.zeros(len(rx_grid))
        if obs["gain_mode"] == "drift":
            gain = (1 if ri == 0 else -1) * rx_grid / 60
        elif obs["gain_mode"] == "step":
            gain = (1 if ri == 0 else -1) * obs["gain_step_db"] * (rx_grid >= cfg["dataset"]["calibration_s"] + (scene.duration - cfg["dataset"]["calibration_s"]) / 2)
        # Ideal main experiment metadata is exactly zero; errors belong to gain tests.
        metadata = gain + (rng.normal(0, obs["gain_metadata_std_db"], len(rx_grid)) if obs["gain_mode"] != "none" else 0)
        jitter = rng.normal(0, obs["jitter_ms"] / 1000, len(rx_grid))
        rx_curves[receiver] = (gain, metadata, jitter)
    for li, link in enumerate(LINKS):
        times = base_times + (0.5 / rate if link[0] == "B" else 0)
        times = times[times < scene.duration]
        rng = generator(cfg, scene.scene_id, f"noise_{link}", noise_seed)
        ideal = channel.response(link, times)
        gain, meta, jitter = [np.interp(times, rx_grid, x) for x in rx_curves[link[1]]]
        common_phase = rng.uniform(-np.pi, np.pi, len(times))
        slope = rng.normal(0, obs["phase_slope_std"], len(times))
        noise_variance = channel.reference_background_power[li] * 10 ** (-obs["snr_db"] / 10)
        noise = np.sqrt(noise_variance / 2) * (rng.standard_normal(ideal.shape) + 1j * rng.standard_normal(ideal.shape))
        csi = 10 ** (gain[:, None] / 20) * np.exp(1j * (common_phase[:, None] + slope[:, None] * channel.k)) * ideal + noise
        valid = generator(cfg, scene.scene_id, f"loss_{link}", noise_seed).random(len(times)) >= obs["loss_rate"]
        if link in obs["outage_links"]:
            valid &= ~((times >= obs["outage_start_s"]) & (times < obs["outage_start_s"] + obs["outage_duration_s"]))
        offset = (1 if link[1] == "C" else -1) * obs["clock_offset_ms"] / 1000
        timestamps = times + jitter + offset
        order = np.argsort(timestamps, kind="stable")
        csi[~valid] = np.nan + 1j * np.nan
        observations[link] = {"timestamp": timestamps[order], "arrival_time": times[order],
                              "packet_id": np.arange(len(times), dtype=np.int64)[order],
                              "csi": csi[order].astype(np.complex64), "valid": valid[order],
                              "gain_db": meta[order].astype(np.float32)}
    return observations
