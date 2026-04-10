"""Streamlit dashboard for secure federated learning demo."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when running: streamlit run dashboard/app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fl_simulation import FLConfig, run_federated_simulation
from security.attack import AttackConfig

st.set_page_config(page_title="SecureFL Dashboard", layout="wide")
st.title("SecureFL: Privacy-Preserving Federated Learning for Medical Data")

with st.sidebar:
    st.header("Simulation Controls")
    num_clients = st.slider("Number of clients", 2, 12, 4)
    num_rounds = st.slider("Rounds", 1, 25, 8)
    local_epochs = st.slider("Local epochs", 1, 10, 2)
    learning_rate = st.slider("Learning rate", 0.001, 0.2, 0.05)
    detection_k = st.slider("Detection threshold k", 0.5, 5.0, 1.0)
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)

    st.subheader("Encryption")
    backend = st.selectbox("Encryption backend", ["simulated", "openfhe"])
    scheme = st.selectbox("OpenFHE scheme", ["bgv", "bfv"], disabled=(backend != "openfhe"))
    enable_encryption_view = st.toggle("Show encryption details", value=True)

    st.subheader("Aggregation & Trust")
    aggregation_method = st.selectbox(
        "Aggregation strategy",
        ["trust_weighted", "fedavg", "trimmed_mean", "coordinate_median"],
    )
    trim_ratio = st.slider(
        "Trim ratio",
        0.0,
        0.45,
        0.2,
        disabled=(aggregation_method != "trimmed_mean"),
    )
    trust_beta = st.slider("Trust smoothing beta", 0.1, 0.95, 0.7)
    trust_min = st.slider("Minimum trust floor", 0.0, 0.5, 0.1)
    trust_penalty = st.slider("Flag penalty", 0.0, 1.0, 0.5)

    st.subheader("Attack Simulation")
    enable_attack = st.toggle("Enable attack simulation", value=True)
    attack_type = st.selectbox("Attack type", ["scaling", "random"], disabled=not enable_attack)
    malicious_fraction = st.slider("Malicious client fraction", 0.0, 0.9, 0.25, disabled=not enable_attack)
    scaling_factor = st.slider(
        "Scaling attack factor",
        1.0,
        20.0,
        8.0,
        disabled=(not enable_attack or attack_type != "scaling"),
    )
    noise_std = st.slider(
        "Random attack noise std",
        0.1,
        10.0,
        5.0,
        disabled=(not enable_attack or attack_type != "random"),
    )

    st.subheader("Data Realism")
    partition_mode = st.selectbox("Client data mode", ["iid", "label_skew"])
    label_skew_strength = st.slider(
        "Label skew strength",
        0.5,
        0.98,
        0.85,
        disabled=(partition_mode != "label_skew"),
    )

    enable_drift = st.toggle("Enable temporal drift", value=False)
    drift_start_round = st.slider("Drift start round", 1, 20, 4, disabled=not enable_drift)
    drift_strength = st.slider("Drift strength", 0.01, 0.5, 0.15, disabled=not enable_drift)

    st.subheader("Privacy & Audit")
    use_dp = st.toggle("Enable DP noise", value=False)
    dp_noise_std = st.slider("DP noise std", 0.001, 0.1, 0.01, disabled=not use_dp)
    enable_audit = st.toggle("Enable audit chain", value=True)

    compare_baseline = st.toggle("Compare against FedAvg baseline", value=True)
    run_clicked = st.button("Run Simulation", type="primary")

if run_clicked:
    if backend == "openfhe" and num_clients != 4:
        st.error("OpenFHE mode in this codebase currently requires exactly 4 clients.")
    else:
        fl_config = FLConfig(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            detection_k=detection_k,
            seed=int(seed),
            encryption_backend=backend,
            encryption_scheme=scheme,
            aggregation_method=aggregation_method,
            trim_ratio=trim_ratio,
            use_dp=use_dp,
            dp_noise_std=dp_noise_std,
            partition_mode=partition_mode,
            label_skew_strength=label_skew_strength,
            enable_drift=enable_drift,
            drift_start_round=drift_start_round,
            drift_strength=drift_strength,
            trust_beta=trust_beta,
            trust_min=trust_min,
            trust_flag_penalty=trust_penalty,
            enable_audit=enable_audit,
        )
        attack_config = AttackConfig(
            enabled=enable_attack,
            attack_type=attack_type,
            malicious_fraction=malicious_fraction,
            scaling_factor=scaling_factor,
            random_noise_std=noise_std,
        )

        with st.spinner("Running federated simulation..."):
            result = run_federated_simulation(fl_config, attack_config)

        baseline_result = None
        if compare_baseline:
            baseline_config = replace(
                fl_config,
                aggregation_method="fedavg",
            )
            with st.spinner("Running FedAvg baseline for comparison..."):
                baseline_result = run_federated_simulation(baseline_config, attack_config)

        rounds_df = pd.DataFrame(result.rounds)

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Accuracy (With Attack)", f"{result.final_accuracy_with_attack:.4f}")
        col2.metric("Final Accuracy (After Filtering)", f"{result.final_accuracy_after_filtering:.4f}")
        col3.metric("Audit Chain Valid", "Yes" if result.audit_chain_valid else "No")

        st.subheader("What Is Unique In This Run")
        summary_df = pd.DataFrame(
            {
                "item": [
                    "Selected strategy",
                    "Detection precision (avg)",
                    "Detection recall (avg)",
                    "Final benign trust (mean)",
                    "Final malicious trust (mean)",
                    "Partition mode",
                    "DP enabled",
                    "Drift enabled",
                ],
                "value": [
                    result.summary.get("selected_strategy"),
                    f"{result.summary.get('mean_detection_precision', 0.0):.3f}",
                    f"{result.summary.get('mean_detection_recall', 0.0):.3f}",
                    f"{result.summary.get('final_mean_trust_benign', 0.0):.3f}",
                    f"{result.summary.get('final_mean_trust_malicious', 0.0):.3f}",
                    result.summary.get("partition_mode"),
                    str(result.summary.get("dp_enabled")),
                    str(result.summary.get("drift_enabled")),
                ],
            }
        )
        st.dataframe(summary_df, width="stretch")

        strategy_note = result.summary.get("strategy_note")
        if strategy_note:
            st.warning(strategy_note)

        if baseline_result is not None:
            delta = result.final_accuracy_selected_strategy - baseline_result.final_accuracy_selected_strategy
            st.info(
                "Selected strategy final accuracy: "
                f"{result.final_accuracy_selected_strategy:.4f} | "
                "FedAvg baseline: "
                f"{baseline_result.final_accuracy_selected_strategy:.4f} | "
                f"Delta: {delta:+.4f}"
            )

        st.subheader("Accuracy Over Rounds")
        st.line_chart(
            rounds_df[["accuracy_with_attack", "accuracy_after_filtering", "accuracy_selected_strategy"]],
            width="stretch",
        )

        st.subheader("Detection Quality Over Rounds")
        det_quality_df = rounds_df[["round_id", "detection_precision", "detection_recall"]].set_index("round_id")
        st.line_chart(det_quality_df, width="stretch")

        st.subheader("Detection Summary")
        detect_df = rounds_df[
            [
                "round_id",
                "detected_clients",
                "malicious_clients_ground_truth",
                "active_clients",
                "threshold",
            ]
        ]
        st.dataframe(detect_df, width="stretch")

        st.subheader("Client Update Distances")
        distance_df = pd.DataFrame(
            {
                "round": rounds_df["round_id"],
                "avg_distance": rounds_df["update_distances"].apply(lambda v: sum(v) / len(v) if v else 0.0),
                "max_distance": rounds_df["update_distances"].apply(lambda v: max(v) if v else 0.0),
            }
        )
        st.bar_chart(distance_df.set_index("round"), width="stretch")

        st.subheader("Trust Evolution")
        trust_df = pd.DataFrame(rounds_df["trust_scores"].tolist())
        trust_df.columns = [f"client_{i}" for i in range(trust_df.shape[1])]
        trust_df.index = rounds_df["round_id"]
        st.line_chart(trust_df, width="stretch")

        if use_dp:
            st.subheader("DP Noise Magnitude")
            dp_df = pd.DataFrame(
                {
                    "round": rounds_df["round_id"],
                    "avg_dp_noise_norm": rounds_df["dp_noise_norms"].apply(lambda v: sum(v) / len(v) if v else 0.0),
                }
            )
            st.bar_chart(dp_df.set_index("round"), width="stretch")

        st.subheader("Audit Chain")
        audit_df = pd.DataFrame(
            {
                "round_id": rounds_df["round_id"],
                "audit_hash": rounds_df["audit_hash"].apply(lambda x: str(x)[:16] + "..." if x != "audit_disabled" else x),
            }
        )
        st.dataframe(audit_df, width="stretch")

        if enable_encryption_view:
            st.subheader("Encrypted Update Preview")
            round_choice = st.slider("Round to inspect", 1, len(rounds_df), 1)
            selected = rounds_df.iloc[round_choice - 1]
            preview_df = pd.DataFrame(
                {
                    "client_id": list(range(len(selected["encrypted_preview"]))),
                    "ciphertext_preview": selected["encrypted_preview"],
                    "update_norm": selected["client_update_norms"],
                    "trust_score": selected["trust_scores"],
                }
            )
            st.dataframe(preview_df, width="stretch")

        st.success("Simulation completed.")
else:
    st.info("Set controls and click 'Run Simulation' to start.")
