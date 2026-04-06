"""End-to-end federated learning simulation orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from core.aggregation import aggregate_updates
from core.model import LogisticRegressionModel
from security.attack import AttackConfig, apply_attack, choose_malicious_clients
from security.audit import AuditRecord, make_round_hash, verify_audit_chain
from security.detection import DetectionResult, detect_malicious_updates
from security.encryption import OpenFHEEncryptionBackend, build_encryption_backend
from security.trust import update_trust_scores
from utils.helpers import (
    apply_feature_drift,
    prepare_medical_demo_data_advanced,
    set_seed,
)


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

    aggregation_method: str = "trust_weighted"  # fedavg | trimmed_mean | coordinate_median | trust_weighted
    trim_ratio: float = 0.2

    use_dp: bool = False
    dp_noise_std: float = 0.01

    partition_mode: str = "iid"  # iid | label_skew
    label_skew_strength: float = 0.85

    enable_drift: bool = False
    drift_start_round: int = 4
    drift_strength: float = 0.15

    trust_beta: float = 0.7
    trust_min: float = 0.1
    trust_flag_penalty: float = 0.5

    enable_audit: bool = True


@dataclass
class RoundLog:
    """Detailed outputs for a single FL round."""

    round_id: int
    accuracy_with_attack: float
    accuracy_after_filtering: float
    accuracy_selected_strategy: float
    strategy_name: str
    detected_clients: list[int]
    malicious_clients_ground_truth: list[int]
    threshold: float
    update_distances: list[float]
    client_update_norms: list[float]
    dp_noise_norms: list[float]
    trust_scores: list[float]
    active_clients: list[int]
    encrypted_preview: list[str]
    detection_precision: float
    detection_recall: float
    audit_hash: str


@dataclass
class SimulationResult:
    """Serializable result payload for CLI and dashboard."""

    config: dict[str, Any]
    rounds: list[dict[str, Any]]
    summary: dict[str, Any]
    final_accuracy_with_attack: float
    final_accuracy_after_filtering: float
    final_accuracy_selected_strategy: float
    audit_chain_valid: bool


def _preview_ciphertext(ciphertext: Any) -> str:
    """Friendly compact representation for dashboard tables."""
    if isinstance(ciphertext, str):
        return ciphertext
    if hasattr(ciphertext, "masked_values"):
        values = np.array(ciphertext.masked_values)[:4]
        return f"sim[{', '.join(f'{v:.4f}' for v in values)} ...]"
    return str(type(ciphertext).__name__)


def _distance_stats(updates: list[np.ndarray]) -> tuple[list[float], float]:
    if not updates:
        return [], 0.0
    stacked = np.vstack(updates)
    mean_update = np.mean(stacked, axis=0)
    distances = np.linalg.norm(stacked - mean_update, axis=1)
    threshold = float(np.mean(distances) + np.std(distances))
    return [float(v) for v in distances], threshold


def _detection_metrics(
    flagged: list[int],
    malicious_truth: list[int],
) -> tuple[float, float]:
    flagged_set = set(flagged)
    truth_set = set(malicious_truth)

    if not truth_set and not flagged_set:
        return 1.0, 1.0
    if not truth_set:
        return 0.0, 1.0

    tp = len(flagged_set & truth_set)
    precision = tp / len(flagged_set) if flagged_set else 0.0
    recall = tp / len(truth_set)
    return float(precision), float(recall)


def run_federated_simulation(config: FLConfig, attack_config: AttackConfig) -> SimulationResult:
    """Run FL training loop with optional attacks, encryption, and filtering."""
    set_seed(config.seed)
    data = prepare_medical_demo_data_advanced(
        num_clients=config.num_clients,
        seed=config.seed,
        partition_mode=config.partition_mode,
        label_skew_strength=config.label_skew_strength,
    )

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

    selected_strategy = config.aggregation_method
    strategy_note = ""
    if isinstance(backend, OpenFHEEncryptionBackend) and selected_strategy != "fedavg":
        # Current OpenFHE wrappers only support additive mean path out-of-the-box.
        strategy_note = "OpenFHE mode currently supports fedavg only; strategy auto-fallback applied."
        selected_strategy = "fedavg"

    trust_scores = np.ones(config.num_clients, dtype=np.float64)
    round_logs: list[RoundLog] = []

    audit_records: list[AuditRecord] = []
    prev_hash = "GENESIS"

    for round_id in range(1, config.num_rounds + 1):
        rng = np.random.default_rng(config.seed + round_id)

        local_updates: list[np.ndarray] = []
        encrypted_updates: list[Any] = []
        dp_noise_norms: list[float] = []

        # 1) local train + update generation
        for client_id, (x_client_base, y_client) in enumerate(data.clients):
            x_client = apply_feature_drift(
                x_client_base,
                client_id=client_id,
                round_id=round_id,
                drift_start_round=config.drift_start_round,
                drift_strength=config.drift_strength if config.enable_drift else 0.0,
                seed=config.seed,
            )

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

            # 3) optional differential privacy perturbation
            if config.use_dp:
                dp_noise = rng.normal(0.0, config.dp_noise_std, size=update.shape)
                update = update + dp_noise
                dp_noise_norms.append(float(np.linalg.norm(dp_noise)))
            else:
                dp_noise_norms.append(0.0)

            local_updates.append(update)

            # 4) encryption
            ciphertext = backend.encrypt_update(update, client_id)
            encrypted_updates.append(ciphertext)

        # 5) baseline aggregation with all updates (attack impact view)
        aggregate_all_cipher = backend.aggregate(encrypted_updates)
        aggregate_all = backend.decrypt_aggregate(
            aggregate_all_cipher,
            participant_count=len(encrypted_updates),
        )
        global_unfiltered = global_unfiltered + aggregate_all

        baseline_model = model_unfiltered.clone()
        baseline_model.set_param_vector(global_unfiltered)
        acc_with_attack = baseline_model.evaluate_accuracy(data.x_test, data.y_test)

        # 6) detection on decrypted client updates
        decrypted_updates = [backend.decrypt_update(c) for c in encrypted_updates]
        distances, fallback_threshold = _distance_stats(decrypted_updates)

        if attack_config.enabled:
            detection: DetectionResult = detect_malicious_updates(decrypted_updates, config.detection_k)
            flagged = detection.flagged_clients
            threshold = detection.threshold
        else:
            flagged = []
            threshold = fallback_threshold

        precision, recall = _detection_metrics(flagged, malicious_clients)

        # 7) filter suspicious clients
        active_ids = [idx for idx in range(config.num_clients) if idx not in flagged]
        if not active_ids:
            active_ids = list(range(config.num_clients))

        # 8) selected strategy aggregation
        if isinstance(backend, OpenFHEEncryptionBackend):
            for dropped_id in flagged:
                backend.overwrite_client_with_zero(dropped_id)

            aggregate_filtered_cipher = backend.aggregate(encrypted_updates)
            aggregate_selected = backend.decrypt_aggregate(
                aggregate_filtered_cipher,
                participant_count=len(active_ids),
            )
        else:
            filtered_updates = [decrypted_updates[idx] for idx in active_ids]
            filtered_trust = [float(trust_scores[idx]) for idx in active_ids]
            aggregate_selected = aggregate_updates(
                filtered_updates,
                method=selected_strategy,
                trim_ratio=config.trim_ratio,
                trust_weights=filtered_trust if selected_strategy == "trust_weighted" else None,
            )

        # 9) update global model using selected strategy
        global_secure = global_secure + aggregate_selected

        secure_model = model_secure.clone()
        secure_model.set_param_vector(global_secure)
        acc_selected = secure_model.evaluate_accuracy(data.x_test, data.y_test)

        # Keep compatibility: 'after_filtering' reflects selected strategy output.
        acc_after_filter = acc_selected

        # 10) adaptive trust update
        trust_scores = update_trust_scores(
            current_trust=trust_scores,
            distances=distances,
            flagged_clients=flagged,
            beta=config.trust_beta,
            min_trust=config.trust_min,
            flag_penalty=config.trust_flag_penalty,
        )

        # 11) audit chain record
        if config.enable_audit:
            round_hash = make_round_hash(
                prev_hash=prev_hash,
                round_id=round_id,
                active_clients=active_ids,
                flagged_clients=flagged,
                aggregated_update=aggregate_selected,
                selected_accuracy=acc_selected,
            )
            audit_records.append(AuditRecord(round_id=round_id, prev_hash=prev_hash, current_hash=round_hash))
            prev_hash = round_hash
        else:
            round_hash = "audit_disabled"

        round_logs.append(
            RoundLog(
                round_id=round_id,
                accuracy_with_attack=acc_with_attack,
                accuracy_after_filtering=acc_after_filter,
                accuracy_selected_strategy=acc_selected,
                strategy_name=selected_strategy,
                detected_clients=flagged,
                malicious_clients_ground_truth=malicious_clients,
                threshold=threshold,
                update_distances=distances,
                client_update_norms=[float(np.linalg.norm(u)) for u in local_updates],
                dp_noise_norms=dp_noise_norms,
                trust_scores=[float(v) for v in trust_scores],
                active_clients=active_ids,
                encrypted_preview=[_preview_ciphertext(c) for c in encrypted_updates],
                detection_precision=precision,
                detection_recall=recall,
                audit_hash=round_hash,
            )
        )

    benign_ids = [idx for idx in range(config.num_clients) if idx not in malicious_clients]
    benign_trust = [float(trust_scores[idx]) for idx in benign_ids] if benign_ids else []
    malicious_trust = [float(trust_scores[idx]) for idx in malicious_clients] if malicious_clients else []

    summary = {
        "selected_strategy": selected_strategy,
        "strategy_note": strategy_note,
        "mean_detection_precision": float(np.mean([r.detection_precision for r in round_logs])) if round_logs else 0.0,
        "mean_detection_recall": float(np.mean([r.detection_recall for r in round_logs])) if round_logs else 0.0,
        "final_mean_trust_benign": float(np.mean(benign_trust)) if benign_trust else 0.0,
        "final_mean_trust_malicious": float(np.mean(malicious_trust)) if malicious_trust else 0.0,
        "audit_enabled": config.enable_audit,
        "partition_mode": config.partition_mode,
        "dp_enabled": config.use_dp,
        "drift_enabled": config.enable_drift,
    }

    return SimulationResult(
        config=asdict(config),
        rounds=[asdict(log) for log in round_logs],
        summary=summary,
        final_accuracy_with_attack=round_logs[-1].accuracy_with_attack if round_logs else 0.0,
        final_accuracy_after_filtering=round_logs[-1].accuracy_after_filtering if round_logs else 0.0,
        final_accuracy_selected_strategy=round_logs[-1].accuracy_selected_strategy if round_logs else 0.0,
        audit_chain_valid=verify_audit_chain(audit_records),
    )
