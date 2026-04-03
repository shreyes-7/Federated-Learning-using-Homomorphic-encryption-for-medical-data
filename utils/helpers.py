"""General helper utilities for dataset prep and reproducibility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    """Container for train/test sets and federated client partitions."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    clients: list[tuple[np.ndarray, np.ndarray]]


def set_seed(seed: int) -> None:
    """Set deterministic NumPy seed."""
    np.random.seed(seed)


def prepare_medical_demo_data(num_clients: int, seed: int, test_size: float = 0.2) -> PreparedData:
    """Load and split sklearn breast-cancer dataset for FL simulation."""
    dataset = load_breast_cancer()
    x = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.float64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clients = split_clients(x_train, y_train, num_clients, seed)
    return PreparedData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        clients=clients,
    )


def split_clients(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split training data into non-overlapping client shards."""
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)

    chunks = np.array_split(indices, num_clients)
    clients: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in chunks:
        clients.append((x_train[chunk], y_train[chunk]))
    return clients
