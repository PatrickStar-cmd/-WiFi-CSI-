"""Known OFDM training, fractional-delay FIR propagation and LS CSI estimates."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .channel import C0, Channel, tones
from .scene import make_scene
from .utils import generator, write_csv, write_json


def fractional_channel(distance, amplitude, cfg, half_length=8):
    radio = cfg["radio"]
    delay = np.asarray(distance) / C0 * radio["sample_rate_hz"]
    integer = np.floor(delay).astype(int)
    fraction = delay - integer
    if np.max(delay) >= radio["cp"]:
        raise ValueError("Propagation delay exceeds the configured CP.")
    if np.max(integer) + half_length > radio["cp"]:
        raise ValueError("Fractional-delay filter needs a longer cyclic prefix for these paths.")
    impulse = np.zeros(int(integer.max()) + 2 * half_length + 1, dtype=complex)
    support = np.arange(-half_length, half_length + 1)
    for d, f, length, a in zip(integer, fraction, distance, amplitude):
        kernel = np.sinc(support - f) * np.kaiser(len(support), 8.6)
        kernel /= kernel.sum()
        coefficient = a * np.exp(-2j * np.pi * radio["fc_hz"] * length / C0)
        impulse[d:d + len(kernel)] += coefficient * kernel
    return impulse, half_length


def training_packet(distance, amplitude, cfg, rng, noise_variance):
    radio = cfg["radio"]
    n, cp, k = radio["nfft"], radio["cp"], tones(cfg)
    bins = k % n
    x_frequency = np.zeros(n, dtype=complex)
    x_frequency[bins] = rng.choice([-1.0, 1.0], len(k))
    x = np.fft.ifft(x_frequency, norm="ortho")
    h, group_delay = fractional_channel(distance, amplitude, cfg)
    # A known cyclic suffix supplies interpolation support at the training boundary.
    # It is a numerical guard, excluded from useful-symbol SNR and rate accounting.
    tx = np.r_[x[-cp:], x, x[:group_delay]]
    clean = np.convolve(tx, h)[cp + group_delay:cp + group_delay + n]
    estimates = []
    for _ in range(2):
        noise = np.sqrt(noise_variance / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        y = np.fft.fft(clean + noise, norm="ortho")
        estimates.append(y[bins] / x_frequency[bins])
    frequencies = radio["fc_hz"] + k * radio["sample_rate_hz"] / n
    ideal = np.sum(np.asarray(amplitude)[:, None] * np.exp(-2j * np.pi * np.asarray(distance)[:, None] * frequencies / C0), axis=0)
    fir_truth = np.sum(h[:, None] * np.exp(-2j * np.pi * (np.arange(len(h)) - group_delay)[:, None] * k / n), axis=0)
    return np.mean(estimates, axis=0), ideal, fir_truth, float(np.mean(abs(clean) ** 2))


def nmse(estimate, truth):
    return float(np.sum(abs(estimate - truth) ** 2) / np.sum(abs(truth) ** 2))


def run_e0(cfg, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scene = make_scene(cfg, 900001, "demo")
    channel = Channel(cfg, scene)
    rng = generator(cfg, scene.scene_id, "ofdm")
    cal = cfg["dataset"]["calibration_s"]
    checks = []
    for kind, center in (("still", cal + 1), ("walk", cal + 7), ("wave", cal + 17)):
        ts = np.linspace(center - 0.25, center + 0.25, 1001)
        positions = scene.positions(ts)
        velocity = np.gradient(positions, ts, axis=0)
        for link in ("AC", "AD", "BC", "BD"):
            d1 = positions - channel.nodes[link[0]]
            d2 = positions - channel.nodes[link[1]]
            directional = np.sum((d1 / np.linalg.norm(d1, axis=-1, keepdims=True) + d2 / np.linalg.norm(d2, axis=-1, keepdims=True)) * velocity, axis=-1)
            expected = -cfg["radio"]["fc_hz"] / C0 * directional
            distance = channel.body_distances(link, ts)
            phase = np.unwrap(np.angle(np.exp(-2j * np.pi * cfg["radio"]["fc_hz"] * distance / C0)), axis=0)
            # exp(-j*2*pi*f*d/c): instantaneous Doppler is phase_rate/(2*pi).
            measured = np.gradient(phase, ts, axis=0) / (2 * np.pi)
            error = float(np.max(abs(expected[2:-2] - measured[2:-2])))
            checks.append({"action": kind, "link": link, "doppler_max_abs_error_hz": error,
                           "max_abs_doppler_hz": float(np.max(abs(expected))),
                           "csi_max_change": float(np.max(abs(channel.response(link, ts) - channel.response(link, [ts[0]]))))})
    boundaries = sorted({s["start"] for s in scene.segments[1:]} | {leg["start"] for s in scene.segments for leg in s["legs"][1:]})
    jumps = [float(np.max(np.linalg.norm(scene.positions([t + 1e-7]) - scene.positions([t - 1e-7]), axis=-1))) for t in boundaries]
    integer_distances = np.array([0, 1, 3]) * C0 / cfg["radio"]["sample_rate_hz"]
    estimate, ideal, _, _ = training_packet(integer_distances, np.array([1, 0.3j, -0.1]), cfg, rng, 0)
    integer_nmse = nmse(estimate, ideal)
    if integer_nmse > 1e-18:
        raise AssertionError(f"Zero-noise integer-delay check failed: NMSE={integer_nmse}")
    if max(jumps, default=0) > 1e-5:
        raise AssertionError("Body trajectory has a discontinuity.")
    if max(row["csi_max_change"] for row in checks if row["action"] == "still") > 1e-10:
        raise AssertionError("Static geometry must generate a static ideal channel.")
    write_json(folder / "physical_checks.json", {"zero_noise_integer_delay_nmse": integer_nmse,
               "trajectory_max_boundary_jump_m": max(jumps, default=0), "checks": checks})
    distances, amplitudes = channel.paths_at("AC", cal + 7)
    records, example = [], None
    for snr in cfg["experiments"]["snr_db"]:
        # Orthonormal FFT + unit BPSK + two independent repeats: sigma_CSI^2=sigma_time^2/2.
        csi_variance = channel.reference_background_power[0] * 10 ** (-snr / 10)
        for packet in range(cfg["experiments"]["ofdm_packets"]):
            estimate, ideal, fir_truth, signal_power = training_packet(distances, amplitudes, cfg, rng, 2 * csi_variance)
            records.append({"csi_snr_db": snr, "packet": packet, "nmse": nmse(estimate, ideal),
                            "noise_nmse_vs_fir": nmse(estimate, fir_truth), "fractional_filter_nmse": nmse(fir_truth, ideal),
                            "waveform_snr_db": 10 * np.log10(signal_power / (2 * csi_variance)),
                            "configured_csi_noise_variance": csi_variance,
                            "measured_csi_noise_power": float(np.mean(abs(estimate - fir_truth) ** 2))})
            if snr == 20 and packet == 0:
                example = (estimate, ideal, fir_truth)
    write_csv(folder / "ofdm_packets.csv", records)
    if example is not None:
        np.savez_compressed(folder / "csi_example.npz", subcarrier=tones(cfg), estimate=example[0], truth=example[1], fir_truth=example[2])
    return records
