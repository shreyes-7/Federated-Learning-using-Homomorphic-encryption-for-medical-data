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


def prepare_medical_demo_data_advanced(
    num_clients: int,
    seed: int,
    partition_mode: str = "iid",
    label_skew_strength: float = 0.85,
    test_size: float = 0.2,
) -> PreparedData:
    """Load and split dataset with selectable client heterogeneity modes."""
    prepared = prepare_medical_demo_data(num_clients=num_clients, seed=seed, test_size=test_size)
    if partition_mode == "iid":
        return prepared
    if partition_mode == "label_skew":
        skewed = split_clients_label_skew(
            prepared.x_train,
            prepared.y_train,
            num_clients=num_clients,
            seed=seed,
            majority_fraction=label_skew_strength,
        )
        prepared.clients = skewed
        return prepared
    raise ValueError(f"Unsupported partition_mode: {partition_mode}")


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


def split_clients_label_skew(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int,
    seed: int,
    majority_fraction: float = 0.85,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create non-IID clients where each client is label-dominant."""
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if not (0.5 <= majority_fraction <= 0.98):
        raise ValueError("majority_fraction must be in [0.5, 0.98]")

    rng = np.random.default_rng(seed)
    idx_class0 = rng.permutation(np.where(y_train == 0)[0]).tolist()
    idx_class1 = rng.permutation(np.where(y_train == 1)[0]).tolist()

    total_samples = len(x_train)
    base_size = total_samples // num_clients
    remainder = total_samples % num_clients
    client_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_clients)]

    clients: list[tuple[np.ndarray, np.ndarray]] = []
    for client_id, csize in enumerate(client_sizes):
        preferred_label = client_id % 2
        preferred_count = int(round(csize * majority_fraction))
        other_count = csize - preferred_count

        preferred_pool = idx_class1 if preferred_label == 1 else idx_class0
        other_pool = idx_class0 if preferred_label == 1 else idx_class1

        take_preferred = min(preferred_count, len(preferred_pool))
        chosen = preferred_pool[:take_preferred]
        del preferred_pool[:take_preferred]

        take_other = min(other_count, len(other_pool))
        chosen.extend(other_pool[:take_other])
        del other_pool[:take_other]

        # Fallback fill from either class if one pool depletes.
        while len(chosen) < csize and (idx_class0 or idx_class1):
            if idx_class0:
                chosen.append(idx_class0.pop(0))
            elif idx_class1:
                chosen.append(idx_class1.pop(0))

        chosen_arr = np.array(chosen, dtype=np.int64)
        rng.shuffle(chosen_arr)
        clients.append((x_train[chosen_arr], y_train[chosen_arr]))

    return clients


def apply_feature_drift(
    x_client: np.ndarray,
    client_id: int,
    round_id: int,
    drift_start_round: int,
    drift_strength: float,
    seed: int,
) -> np.ndarray:
    """Inject deterministic feature drift to emulate site/device protocol shifts."""
    if round_id < drift_start_round or drift_strength <= 0.0:
        return x_client

    rng = np.random.default_rng(seed + client_id * 97 + round_id * 13)
    drifted = x_client.copy()
    n_features = drifted.shape[1]
    n_shift = max(1, int(0.25 * n_features))
    cols = rng.choice(n_features, size=n_shift, replace=False)

    # Progressive drift as rounds advance beyond drift_start_round.
    round_factor = 1.0 + 0.1 * (round_id - drift_start_round)
    shift = rng.normal(loc=drift_strength * round_factor, scale=0.05, size=n_shift)
    drifted[:, cols] += shift
    return drifted
