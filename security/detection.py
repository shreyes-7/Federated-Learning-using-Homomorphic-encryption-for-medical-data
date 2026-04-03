"""Anomaly detection for client model updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """Detection outputs for one FL round."""

    flagged_clients: list[int]
    distances: list[float]
    threshold: float


def detect_malicious_updates(updates: list[np.ndarray], k: float) -> DetectionResult:
    """
    Detect outlier updates using L2-distance from mean update.

    threshold = mean(distance) + k * std(distance)
    """
    if not updates:
        return DetectionResult(flagged_clients=[], distances=[], threshold=0.0)

    stacked = np.vstack(updates)
    mean_update = np.mean(stacked, axis=0)
    distances = np.linalg.norm(stacked - mean_update, axis=1)

    mu = float(np.mean(distances))
    sigma = float(np.std(distances))
    threshold = mu + k * sigma

    flagged = [idx for idx, d in enumerate(distances) if d > threshold]
    return DetectionResult(
        flagged_clients=flagged,
        distances=[float(v) for v in distances],
        threshold=float(threshold),
    )
