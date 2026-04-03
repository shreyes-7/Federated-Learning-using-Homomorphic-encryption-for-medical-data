"""Aggregation helpers for federated updates."""

from __future__ import annotations

import numpy as np


def mean_aggregate(updates: list[np.ndarray]) -> np.ndarray:
    """Aggregate client updates by arithmetic mean."""
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")
    stacked = np.vstack(updates)
    return np.mean(stacked, axis=0)


def apply_global_update(global_params: np.ndarray, aggregated_update: np.ndarray) -> np.ndarray:
    """Apply an aggregated update and return updated global parameters."""
    return global_params + aggregated_update
