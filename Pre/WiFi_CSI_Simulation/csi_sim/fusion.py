"""Observation-quality fusion and causal state confirmation."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .config import LINKS

TOPOLOGIES = {"AC": [0], "AD": [1], "BC": [2], "BD": [3], "dual_AC_AD": [0, 1], "four_equal": [0, 1, 2, 3], "four_quality": [0, 1, 2, 3]}


def fuse(scores, valid, quality, topology, ratio_limit=10.0, score_domain="linear"):
    indices = TOPOLOGIES[topology]
    good = valid[:, indices] & np.isfinite(scores[:, indices])
    weights = np.where(good, np.asarray(quality[:, indices], dtype=np.float64) if topology == "four_quality" else 1.0, 0.0)
    if topology == "four_quality":
        minimum = np.min(np.where(weights > 0, weights, np.inf), axis=1)
        weights = np.minimum(weights, minimum[:, None] * ratio_limit)
    denominator = weights.sum(axis=1)
    values = np.where(good, scores[:, indices], 0.0)
    probability = np.full(len(scores), np.nan)
    ok = denominator > 0
    if score_domain == "negative_log_static_probability":
        # e=-log(1-p). This is exactly weighted probability averaging, computed
        # through its complement so that probabilities near one remain ordered.
        log_weights = np.full(weights.shape, -np.inf)
        positive = weights > 0
        log_weights[positive] = np.log(weights[positive])
        probability[ok] = np.maximum(0.0, np.log(denominator[ok]) - logsumexp(log_weights[ok] - values[ok], axis=1))
    elif score_domain == "linear":
        probability[ok] = np.sum(weights * values, axis=1)[ok] / denominator[ok]
    else:
        raise ValueError(score_domain)
    return probability, good.sum(axis=1)


def confirm_states(scores, threshold, count=2):
    """Static calibration initializes state=0; invalid data resets confirmation."""
    scores = np.asarray(scores)
    thresholds = np.atleast_1d(threshold)
    output = np.full((len(scores), len(thresholds)), -1, dtype=np.int8)
    state = np.zeros(len(thresholds), dtype=np.int8)
    pending = np.full(len(thresholds), -1, dtype=np.int8)
    streak = np.zeros(len(thresholds), dtype=int)
    for i, score in enumerate(scores):
        if not np.isfinite(score):
            state[:], pending[:], streak[:] = -1, -1, 0
            continue
        candidate = (score >= thresholds).astype(np.int8)
        streak = np.where(candidate == pending, streak + 1, 1)
        pending = candidate
        state = np.where(streak >= count, candidate, state)
        output[i] = state
    return output[:, 0] if np.ndim(threshold) == 0 else output


def select_threshold(sequences, cfg):
    """Choose using post-confirmation validation FPR; endpoints include constant decisions."""
    det = cfg["detector"]
    finite = np.concatenate([s["score"][np.isfinite(s["score"])] for s in sequences])
    if finite.size == 0:
        raise ValueError("Validation has no finite scores.")
    candidates = np.unique(np.r_[np.nextafter(finite.min(), -np.inf),
                                  np.quantile(finite, np.linspace(0, 1, det["threshold_candidates"])),
                                  np.nextafter(finite.max(), np.inf)])
    tp, fp, fn, tn = (np.zeros(len(candidates), dtype=np.int64) for _ in range(4))
    for sequence in sequences:
        states = confirm_states(sequence["score"], candidates, det["confirmation_count"])
        y = np.asarray(sequence["truth"])[:, None]
        tp += ((states == 1) & (y == 1)).sum(axis=0)
        fp += ((states == 1) & (y == 0)).sum(axis=0)
        fn += ((states == 0) & (y == 1)).sum(axis=0)
        tn += ((states == 0) & (y == 0)).sum(axis=0)
    fpr = fp / np.maximum(fp + tn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    allowed = np.flatnonzero(fpr <= det["validation_fpr_limit"] + 1e-12)
    # A threshold above all finite scores always provides an all-static candidate.
    best = max(allowed, key=lambda i: (f1[i], candidates[i]))
    return {"threshold": float(candidates[best]), "validation_f1": float(f1[best]),
            "validation_fpr": float(fpr[best]), "candidate_count": len(candidates)}


def resources(topology, rate, duration, cfg):
    indices = TOPOLOGIES[topology]
    links = [LINKS[i] for i in indices]
    senders, receivers = set(link[0] for link in links), set(link[1] for link in links)
    count = 0
    for sender in senders:
        offset = 0.5 / rate if sender == "B" else 0
        count += len(np.arange(offset, duration, 1 / rate))
    return {"link_count": len(links), "receiver_count": len(receivers), "transmitter_count": len(senders),
            "total_packets": count, "transmit_pps": rate * len(senders), "link_rate_hz": rate,
            "tx_power_dbm": cfg["radio"]["tx_power_dbm"], "packet_bytes": cfg["radio"]["packet_bytes"],
            "phy_rate_mbps": cfg["radio"]["phy_rate_mbps"]}
