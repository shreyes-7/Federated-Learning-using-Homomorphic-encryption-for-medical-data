"""CLI entrypoint for secure federated learning simulation."""

from __future__ import annotations

import argparse
import json

from core.fl_simulation import FLConfig, run_federated_simulation
from security.attack import AttackConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Secure Federated Learning simulator")
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--detection-k", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backend", choices=["simulated", "openfhe"], default="simulated")
    parser.add_argument("--scheme", choices=["bgv", "bfv"], default="bgv")
    parser.add_argument(
        "--aggregation-method",
        choices=["fedavg", "trimmed_mean", "coordinate_median", "trust_weighted"],
        default="trust_weighted",
    )
    parser.add_argument("--trim-ratio", type=float, default=0.2)

    parser.add_argument("--partition-mode", choices=["iid", "label_skew"], default="iid")
    parser.add_argument("--label-skew-strength", type=float, default=0.85)

    parser.add_argument("--enable-drift", action="store_true")
    parser.add_argument("--drift-start-round", type=int, default=4)
    parser.add_argument("--drift-strength", type=float, default=0.15)

    parser.add_argument("--dp", action="store_true", help="Enable DP noise on client updates")
    parser.add_argument("--dp-noise-std", type=float, default=0.01)

    parser.add_argument("--trust-beta", type=float, default=0.7)
    parser.add_argument("--trust-min", type=float, default=0.1)
    parser.add_argument("--trust-flag-penalty", type=float, default=0.5)
    parser.add_argument("--disable-audit", action="store_true")

    parser.add_argument("--attack", action="store_true", help="Enable malicious attack simulation")
    parser.add_argument("--attack-type", choices=["scaling", "random"], default="scaling")
    parser.add_argument("--malicious-fraction", type=float, default=0.25)
    parser.add_argument("--scaling-factor", type=float, default=8.0)
    parser.add_argument("--noise-std", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    fl_config = FLConfig(
        num_clients=args.num_clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        detection_k=args.detection_k,
        seed=args.seed,
        encryption_backend=args.backend,
        encryption_scheme=args.scheme,
        aggregation_method=args.aggregation_method,
        trim_ratio=args.trim_ratio,
        use_dp=args.dp,
        dp_noise_std=args.dp_noise_std,
        partition_mode=args.partition_mode,
        label_skew_strength=args.label_skew_strength,
        enable_drift=args.enable_drift,
        drift_start_round=args.drift_start_round,
        drift_strength=args.drift_strength,
        trust_beta=args.trust_beta,
        trust_min=args.trust_min,
        trust_flag_penalty=args.trust_flag_penalty,
        enable_audit=not args.disable_audit,
    )

    attack_config = AttackConfig(
        enabled=args.attack,
        attack_type=args.attack_type,
        malicious_fraction=args.malicious_fraction,
        scaling_factor=args.scaling_factor,
        random_noise_std=args.noise_std,
    )

    result = run_federated_simulation(fl_config, attack_config)

    print("=== SecureFL Simulation Summary ===")
    print(f"Backend: {args.backend} ({args.scheme})")
    print(f"Strategy: {result.summary.get('selected_strategy')}")
    print(f"Final accuracy with attack: {result.final_accuracy_with_attack:.4f}")
    print(f"Final accuracy after filtering: {result.final_accuracy_after_filtering:.4f}")
    print(f"Audit chain valid: {result.audit_chain_valid}")
    if result.summary.get("strategy_note"):
        print(f"Note: {result.summary['strategy_note']}")
    print("Summary:")
    print(json.dumps(result.summary, indent=2))
    print("\nRound details:")
    print(json.dumps(result.rounds, indent=2))


if __name__ == "__main__":
    main()
