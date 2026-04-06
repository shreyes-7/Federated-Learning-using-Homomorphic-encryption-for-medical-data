"""Aggregation helpers for federated updates."""

from __future__ import annotations

import numpy as np


def mean_aggregate(updates: list[np.ndarray]) -> np.ndarray:
    """Aggregate client updates by arithmetic mean."""
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    stacked = np.vstack(updates)
    return np.mean(stacked, axis=0)


def trimmed_mean_aggregate(updates: list[np.ndarray], trim_ratio: float = 0.2) -> np.ndarray:
    """Coordinate-wise trimmed mean aggregation."""
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0.0, 0.5)")

    stacked = np.vstack(updates)
    n_clients = stacked.shape[0]
    trim_k = int(np.floor(n_clients * trim_ratio))
    if n_clients - 2 * trim_k <= 0:
        return np.mean(stacked, axis=0)

    sorted_vals = np.sort(stacked, axis=0)
    trimmed = sorted_vals[trim_k : n_clients - trim_k, :]
    return np.mean(trimmed, axis=0)


def coordinate_median_aggregate(updates: list[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median aggregation."""
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    stacked = np.vstack(updates)
    return np.median(stacked, axis=0)


def trust_weighted_aggregate(updates: list[np.ndarray], trust_weights: list[float]) -> np.ndarray:
    """Weighted aggregation using client trust scores."""
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    if len(updates) != len(trust_weights):
        raise ValueError("updates and trust_weights must have same length")

    stacked = np.vstack(updates)
    weights = np.array(trust_weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    weights /= np.sum(weights)
    return np.sum(stacked * weights[:, None], axis=0)


def aggregate_updates(
    updates: list[np.ndarray],
    method: str,
    trim_ratio: float = 0.2,
    trust_weights: list[float] | None = None,
) -> np.ndarray:
    """Route to selected robust aggregation strategy."""
    method = method.lower()
    if method == "fedavg":
        return mean_aggregate(updates)
    if method == "trimmed_mean":
        return trimmed_mean_aggregate(updates, trim_ratio=trim_ratio)
    if method == "coordinate_median":
        return coordinate_median_aggregate(updates)
    if method == "trust_weighted":
        if trust_weights is None:
            raise ValueError("trust_weights required for trust_weighted aggregation")
        return trust_weighted_aggregate(updates, trust_weights=trust_weights)
    raise ValueError(f"Unsupported aggregation method: {method}")


def apply_global_update(global_params: np.ndarray, aggregated_update: np.ndarray) -> np.ndarray:
    """Apply an aggregated update and return updated global parameters."""
    return global_params + aggregated_update
