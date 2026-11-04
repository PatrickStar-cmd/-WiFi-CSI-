"""All detector features are calculated exclusively from observations."""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

FEATURE_NAMES = [f"{name}_{stat}" for name in ("amplitude_variance", "derivative_energy", "power_acf1", "band_0p5_3", "band_3_10", "band_10_40") for stat in ("median", "iqr")] + [f"frequency_group_{i}_variance" for i in range(4)]


def power_acf(amplitude, packet_id, epsilon):
    power = amplitude ** 2
    centered = power - power.mean(axis=0)
    pairs = np.diff(packet_id) == 1
    if pairs.sum() < 2:
        raise ValueError("A valid window needs two original adjacent-packet pairs.")
    covariance = np.mean((centered[:-1] * centered[1:])[pairs], axis=0)
    return np.clip(covariance / (np.mean(centered ** 2, axis=0) + epsilon), -1, 1)


def compute_features(raw_amplitude, raw_times, packet_id, filled_amplitude, rate, epsilon, rssi_offset_db):
    variance = raw_amplitude.var(axis=0)
    dt = np.diff(raw_times)
    adjacent = (np.diff(packet_id) == 1) & (dt > 1e-8)
    if adjacent.sum() < 2:
        raise ValueError("Insufficient original adjacent packets for derivative features.")
    derivative = np.mean((np.diff(raw_amplitude, axis=0)[adjacent] / dt[adjacent, None]) ** 2, axis=0)
    acf = power_acf(raw_amplitude, packet_id, epsilon)
    freq, spectrum = periodogram(filled_amplitude, fs=rate, window="hann", detrend="constant", axis=0, scaling="density")
    total = spectrum[freq >= 0.5].sum(axis=0) + epsilon
    bands = []
    for lo, hi in ((0.5, 3), (3, 10), (10, 40)):
        selected = (freq >= lo) & ((freq < hi) if hi != 40 else (freq <= hi))
        bands.append(spectrum[selected].sum(axis=0) / total)
    values = []
    for feature in (variance, derivative, acf, *bands):
        values.extend([np.median(feature), np.quantile(feature, 0.75) - np.quantile(feature, 0.25)])
    values.extend(np.var(raw_amplitude[:, indices].mean(axis=1)) for indices in np.array_split(np.arange(raw_amplitude.shape[1]), 4))
    rssi = 10 * np.log10(np.mean(raw_amplitude ** 2, axis=1) + epsilon) + rssi_offset_db
    baselines = [np.var(rssi), np.var(np.round(rssi)), np.median(variance), np.mean(acf)]
    return np.asarray(values, dtype=np.float32), np.asarray(baselines, dtype=np.float32)
