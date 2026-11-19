"""Causal P2 confirmation when P1 is already classified as moving."""

from dataclasses import replace

import numpy as np
from threadpoolctl import threadpool_limits

from . import multi_detector_v2 as v2
from .multi_detector import MIN_LINKS, geometry_config


CONFIRM_THRESHOLD = .24
BYPASS_THRESHOLD = .85
CONFIRM_WINDOWS = 3


def calibrate(config, progress=None):
    calibration = v2.calibrate(config, progress)
    return replace(calibration, metadata={
        **calibration.metadata,
        'method': 'mixed-amplitude separate motion heads with causal P2 confirmation',
        'p2Confirmation': {
            'condition': 'P1 inferred moving, P2 not yet active',
            'scoreThreshold': CONFIRM_THRESHOLD,
            'windows': CONFIRM_WINDOWS,
            'bypassThreshold': BYPASS_THRESHOLD,
        },
    })


def _causal_joint_scores(scores, valid):
    posterior = np.full((len(scores), 4), np.nan)
    active = np.zeros(2, dtype=bool)
    evidence = 0
    for index, raw in enumerate(scores):
        if not valid[index]:
            active[:] = False
            evidence = 0
            continue
        p1, p2 = raw
        boundary = np.where(active, v2.CONTINUE_THRESHOLDS, v2.START_THRESHOLDS)
        if p1 >= boundary[0]:
            evidence = evidence + 1 if p2 >= CONFIRM_THRESHOLD else 0
            if not active[1]:
                boundary[1] = CONFIRM_THRESHOLD if evidence >= CONFIRM_WINDOWS else BYPASS_THRESHOLD
        else:
            evidence = 0
        scaled = raw * (1 - boundary)
        p, q = scaled / (scaled + (1 - raw) * boundary)
        active[:] = (p >= .5, q >= .5)
        posterior[index] = ((1 - p) * (1 - q), p * (1 - q), (1 - p) * q, p * q)
    return posterior


def predict(observations, rate, names, times, calibration):
    """Infer each person from past and current mixed CSI only."""
    if tuple(names) != calibration.names or rate != calibration.rate:
        raise ValueError('链路或采样率与双人校准不一致。')
    features, valid_links = v2._observable_features(observations, rate, names, times)
    valid = valid_links.sum(axis=1) >= MIN_LINKS
    scores = np.full((len(times), 2), np.nan)
    if valid.any():
        with threadpool_limits(limits=2):
            scores[valid] = np.column_stack([head.predict_proba(features[valid])[:, 1] for head in calibration.heads])
    posterior = _causal_joint_scores(scores, valid)
    return {'posterior': posterior, 'valid': valid, 'confidence': posterior.max(axis=1), 'valid_links': valid_links}
