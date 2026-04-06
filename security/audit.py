"""Lightweight verifiable audit chain for simulation rounds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class AuditRecord:
    """Single round audit proof metadata."""

    round_id: int
    prev_hash: str
    current_hash: str


def _hash_payload(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def make_round_hash(
    prev_hash: str,
    round_id: int,
    active_clients: list[int],
    flagged_clients: list[int],
    aggregated_update: np.ndarray,
    selected_accuracy: float,
) -> str:
    """Create deterministic hash for one FL round state transition."""
    payload = {
        "prev_hash": prev_hash,
        "round_id": round_id,
        "active_clients": active_clients,
        "flagged_clients": flagged_clients,
        "agg_digest": hashlib.sha256(
            np.ascontiguousarray(np.round(aggregated_update, 8), dtype=np.float64).tobytes()
        ).hexdigest(),
        "selected_accuracy": round(float(selected_accuracy), 8),
    }
    return _hash_payload(payload)


def verify_audit_chain(records: list[AuditRecord]) -> bool:
    """Check prev-hash linkage continuity."""
    if not records:
        return True
    if records[0].prev_hash != "GENESIS":
        return False
    for idx in range(1, len(records)):
        if records[idx].prev_hash != records[idx - 1].current_hash:
            return False
    return True
