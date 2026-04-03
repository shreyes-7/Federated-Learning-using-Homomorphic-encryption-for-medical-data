"""Attack simulation utilities for malicious federated clients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AttackConfig:
    """Configuration for malicious update injection."""

    enabled: bool = False
    attack_type: str = "scaling"  # scaling | random
    malicious_fraction: float = 0.25
    scaling_factor: float = 8.0
    random_noise_std: float = 5.0


def choose_malicious_clients(num_clients: int, fraction: float, seed: int) -> list[int]:
    """Select a deterministic set of malicious client ids."""
    if num_clients <= 0:
        return []
    count = max(1, int(round(num_clients * fraction))) if fraction > 0 else 0
    count = min(count, num_clients)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(num_clients, size=count, replace=False).tolist())


def apply_attack(update: np.ndarray, config: AttackConfig, rng: np.random.Generator) -> np.ndarray:
    """Transform update according to selected attack type."""
    if config.attack_type == "scaling":
        return update * config.scaling_factor
    if config.attack_type == "random":
        return update + rng.normal(0.0, config.random_noise_std, size=update.shape)
    raise ValueError(f"Unsupported attack type: {config.attack_type}")
