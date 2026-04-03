"""End-to-end federated learning simulation orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from core.model import LogisticRegressionModel
from security.attack import AttackConfig, apply_attack, choose_malicious_clients
from security.detection import DetectionResult, detect_malicious_updates
from security.encryption import OpenFHEEncryptionBackend, build_encryption_backend
from utils.helpers import prepare_medical_demo_data, set_seed


@dataclass
class FLConfig:
    """Core simulation configuration."""

    num_clients: int = 4
    num_rounds: int = 10
    local_epochs: int = 2
    learning_rate: float = 0.05
    detection_k: float = 1.0
    seed: int = 42
    encryption_backend: str = "simulated"  # simulated | openfhe
    encryption_scheme: str = "bgv"  # used only for openfhe


@dataclass
class RoundLog:
    """Detailed outputs for a single FL round."""

    round_id: int
    accuracy_with_attack: float
    accuracy_after_filtering: float
    detected_clients: list[int]
    malicious_clients_ground_truth: list[int]
    threshold: float
    update_distances: list[float]
    client_update_norms: list[float]
    encrypted_preview: list[str]


@dataclass
class SimulationResult:
    """Serializable result payload for CLI and dashboard."""

    config: dict[str, Any]
    rounds: list[dict[str, Any]]
    final_accuracy_with_attack: float
    final_accuracy_after_filtering: float


def _preview_ciphertext(ciphertext: Any) -> str:
    """Friendly compact representation for dashboard tables."""
    if isinstance(ciphertext, str):
        return ciphertext
    if hasattr(ciphertext, "masked_values"):
        values = np.array(ciphertext.masked_values)[:4]
        return f"sim[{', '.join(f'{v:.4f}' for v in values)} ...]"
    return str(type(ciphertext).__name__)


def run_federated_simulation(config: FLConfig, attack_config: AttackConfig) -> SimulationResult:
    """Run FL training loop with optional attacks, encryption, and filtering."""
    set_seed(config.seed)
    data = prepare_medical_demo_data(config.num_clients, config.seed)

    model_secure = LogisticRegressionModel(n_features=data.x_train.shape[1])
    model_unfiltered = model_secure.clone()

    global_secure = model_secure.get_param_vector()
    global_unfiltered = model_unfiltered.get_param_vector()

    backend = build_encryption_backend(
        backend_type=config.encryption_backend,
        scheme=config.encryption_scheme,
        seed=config.seed,
    )
    backend.setup(config.num_clients, vector_dim=global_secure.shape[0])
    if isinstance(backend, OpenFHEEncryptionBackend):
        backend.ensure_required_paths()

    malicious_clients = (
        choose_malicious_clients(config.num_clients, attack_config.malicious_fraction, config.seed)
        if attack_config.enabled
        else []
    )

    round_logs: list[RoundLog] = []

    for round_id in range(1, config.num_rounds + 1):
        rng = np.random.default_rng(config.seed + round_id)

        local_updates: list[np.ndarray] = []
        encrypted_updates: list[Any] = []

        # 1) local train + update generation
        for client_id, (x_client, y_client) in enumerate(data.clients):
            client_model = model_secure.clone()
            client_model.set_param_vector(global_secure)
            client_model.train_local(
                x_client,
                y_client,
                epochs=config.local_epochs,
                lr=config.learning_rate,
            )

            local_params = client_model.get_param_vector()
            update = local_params - global_secure

            # 2) optional attack
            if attack_config.enabled and client_id in malicious_clients:
                update = apply_attack(update, attack_config, rng)

            local_updates.append(update)

            # 3) encryption
            ciphertext = backend.encrypt_update(update, client_id)
            encrypted_updates.append(ciphertext)

        # 4) build attack-only baseline (no filtering)
        aggregate_all_cipher = backend.aggregate(encrypted_updates)
        aggregate_all = backend.decrypt_aggregate(
            aggregate_all_cipher,
            participant_count=len(encrypted_updates),
        )
        global_unfiltered = global_unfiltered + aggregate_all

        baseline_model = model_unfiltered.clone()
        baseline_model.set_param_vector(global_unfiltered)
        acc_with_attack = baseline_model.evaluate_accuracy(data.x_test, data.y_test)

        # 5) anomaly detection on decrypted encrypted client updates
        if attack_config.enabled:
            decrypted_for_detection = [backend.decrypt_update(c) for c in encrypted_updates]
            detection: DetectionResult = detect_malicious_updates(
                decrypted_for_detection, config.detection_k
            )
            flagged = detection.flagged_clients
        else:
            detection = DetectionResult(
                flagged_clients=[],
                distances=[0.0 for _ in range(config.num_clients)],
                threshold=0.0,
            )
            flagged = []

        # 6) filter flagged clients before secure aggregation
        active_ids = [idx for idx in range(config.num_clients) if idx not in flagged]
        if not active_ids:
            active_ids = list(range(config.num_clients))

        if isinstance(backend, OpenFHEEncryptionBackend):
            # OpenFHE server currently always aggregates 4 fixed files.
            # Dropped clients are replaced with encrypted zero updates.
            for dropped_id in flagged:
                backend.overwrite_client_with_zero(dropped_id)
            filtered_ciphertexts = encrypted_updates
        else:
            filtered_ciphertexts = [encrypted_updates[idx] for idx in active_ids]

        aggregate_filtered_cipher = backend.aggregate(filtered_ciphertexts)
        aggregate_filtered = backend.decrypt_aggregate(
            aggregate_filtered_cipher,
            participant_count=len(active_ids),
        )

        global_secure = global_secure + aggregate_filtered

        secure_model = model_secure.clone()
        secure_model.set_param_vector(global_secure)
        acc_after_filter = secure_model.evaluate_accuracy(data.x_test, data.y_test)

        round_logs.append(
            RoundLog(
                round_id=round_id,
                accuracy_with_attack=acc_with_attack,
                accuracy_after_filtering=acc_after_filter,
                detected_clients=flagged,
                malicious_clients_ground_truth=malicious_clients,
                threshold=detection.threshold,
                update_distances=detection.distances,
                client_update_norms=[float(np.linalg.norm(u)) for u in local_updates],
                encrypted_preview=[_preview_ciphertext(c) for c in encrypted_updates],
            )
        )

    return SimulationResult(
        config=asdict(config),
        rounds=[asdict(log) for log in round_logs],
        final_accuracy_with_attack=round_logs[-1].accuracy_with_attack if round_logs else 0.0,
        final_accuracy_after_filtering=round_logs[-1].accuracy_after_filtering if round_logs else 0.0,
    )
