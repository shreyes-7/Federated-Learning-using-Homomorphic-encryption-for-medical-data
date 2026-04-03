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
    print(f"Final accuracy with attack: {result.final_accuracy_with_attack:.4f}")
    print(f"Final accuracy after filtering: {result.final_accuracy_after_filtering:.4f}")
    print("\nRound details:")
    print(json.dumps(result.rounds, indent=2))


if __name__ == "__main__":
    main()
