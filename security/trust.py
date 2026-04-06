"""Adaptive client trust scoring for secure federated aggregation."""

from __future__ import annotations

import numpy as np


def update_trust_scores(
    current_trust: np.ndarray,
    distances: list[float],
    flagged_clients: list[int],
    beta: float,
    min_trust: float,
    flag_penalty: float,
) -> np.ndarray:
    """
    Update trust scores using distance consistency and anomaly flags.

    Higher distance => lower trust contribution.
    Flagged clients receive additional multiplicative penalty.
    """
    if len(current_trust) != len(distances):
        raise ValueError("Trust and distance length mismatch")

    distances_arr = np.array(distances, dtype=np.float64)
    # Robust scale to normalize distance impact across rounds.
    scale = float(np.median(distances_arr) + np.std(distances_arr) + 1e-8)
    instant_score = np.exp(-distances_arr / scale)

    updated = beta * current_trust + (1.0 - beta) * instant_score

    if flagged_clients:
        updated[np.array(flagged_clients, dtype=np.int32)] *= max(0.0, 1.0 - flag_penalty)

    return np.clip(updated, min_trust, 1.0)
