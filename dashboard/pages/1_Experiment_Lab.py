"""Experiment Lab page for batch simulation comparisons."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fl_simulation import FLConfig, run_federated_simulation
from security.attack import AttackConfig

st.set_page_config(page_title="Experiment Lab", layout="wide")
st.title("Experiment Lab")
st.info(
    "Run a small matrix of experiments automatically to compare aggregation choices "
    "and the impact of attacks on final performance and robustness."
)


def _build_rounds_frame(result, label: str) -> pd.DataFrame:
    rounds_df = pd.DataFrame(result.rounds)
    rounds_df["experiment"] = label
    return rounds_df


with st.sidebar:
    st.header("Batch Settings")
    num_clients = st.slider("Number of clients", 2, 12, 6)
    num_rounds = st.slider("Rounds", 1, 25, 8)
    local_epochs = st.slider("Local epochs", 1, 10, 2)
    learning_rate = st.slider("Learning rate", 0.001, 0.2, 0.05)
    detection_k = st.slider("Detection threshold k", 0.5, 5.0, 1.0)
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)

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

    st.subheader("Attack Settings")
    attack_type = st.selectbox("Attack type", ["scaling", "random"])
    malicious_fraction = st.slider("Malicious client fraction", 0.0, 0.9, 0.25)
    scaling_factor = st.slider(
        "Scaling attack factor",
        1.0,
        20.0,
        8.0,
        disabled=(attack_type != "scaling"),
    )
    noise_std = st.slider(
        "Random attack noise std",
        0.1,
        10.0,
        5.0,
        disabled=(attack_type != "random"),
    )

    run_clicked = st.button("Run Experiment Matrix", type="primary")

if run_clicked:
    base_config = FLConfig(
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        detection_k=detection_k,
        seed=int(seed),
        encryption_backend="simulated",
        encryption_scheme="bgv",
        trim_ratio=0.2,
        partition_mode=partition_mode,
        label_skew_strength=label_skew_strength,
        enable_drift=enable_drift,
        drift_start_round=drift_start_round,
        drift_strength=drift_strength,
        use_dp=use_dp,
        dp_noise_std=dp_noise_std,
        enable_audit=enable_audit,
    )

    experiments = [
        ("fedavg | no attack", "fedavg", False),
        ("fedavg | attack", "fedavg", True),
        ("trust_weighted | no attack", "trust_weighted", False),
        ("trust_weighted | attack", "trust_weighted", True),
    ]

    summary_rows: list[dict[str, object]] = []
    rounds_frames: list[pd.DataFrame] = []
    download_payload: list[dict[str, object]] = []

    progress = st.progress(0.0, text="Preparing experiments...")
    for idx, (label, strategy, attack_enabled) in enumerate(experiments, start=1):
        config = FLConfig(**{**asdict(base_config), "aggregation_method": strategy})
        attack_config = AttackConfig(
            enabled=attack_enabled,
            attack_type=attack_type,
            malicious_fraction=malicious_fraction,
            scaling_factor=scaling_factor,
            random_noise_std=noise_std,
        )

        progress.progress(idx / len(experiments), text=f"Running {label}")
        result = run_federated_simulation(config, attack_config)
        rounds_frames.append(_build_rounds_frame(result, label))

        robustness_gain = result.final_accuracy_after_filtering - result.final_accuracy_with_attack
        summary_rows.append(
            {
                "experiment": label,
                "strategy": strategy,
                "attack_enabled": attack_enabled,
                "final_accuracy_with_attack": result.final_accuracy_with_attack,
                "final_accuracy_after_filtering": result.final_accuracy_after_filtering,
                "final_accuracy_selected_strategy": result.final_accuracy_selected_strategy,
                "mean_detection_precision": result.summary.get("mean_detection_precision", 0.0),
                "mean_detection_recall": result.summary.get("mean_detection_recall", 0.0),
                "robustness_gain": robustness_gain,
                "final_mean_trust_benign": result.summary.get("final_mean_trust_benign", 0.0),
                "final_mean_trust_malicious": result.summary.get("final_mean_trust_malicious", 0.0),
            }
        )
        download_payload.append(
            {
                "experiment": label,
                "config": asdict(config),
                "attack_config": asdict(attack_config),
                "summary": result.summary,
                "rounds": result.rounds,
            }
        )

    progress.empty()

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="final_accuracy_selected_strategy",
        ascending=False,
    )
    all_rounds_df = pd.concat(rounds_frames, ignore_index=True)

    best_row = summary_df.iloc[0]
    st.subheader("Key Takeaways")
    st.success(
        f"Best performing setup: {best_row['experiment']} "
        f"with final selected-strategy accuracy {best_row['final_accuracy_selected_strategy']:.4f}."
    )
    st.markdown(
        "- `trust_weighted` usually helps most when attacks are enabled because low-trust clients get down-weighted.\n"
        "- `robustness_gain` shows how much defended performance improved over the attacked baseline."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Accuracy Comparison")
        accuracy_chart = summary_df.set_index("experiment")[
            [
                "final_accuracy_with_attack",
                "final_accuracy_after_filtering",
                "final_accuracy_selected_strategy",
            ]
        ]
        st.bar_chart(accuracy_chart, width="stretch")

    with col2:
        st.subheader("Robustness And Detection")
        robustness_chart = summary_df.set_index("experiment")[
            [
                "robustness_gain",
                "mean_detection_precision",
                "mean_detection_recall",
            ]
        ]
        st.bar_chart(robustness_chart, width="stretch")

    st.subheader("Round-wise Accuracy")
    accuracy_over_time = all_rounds_df.pivot(
        index="round_id",
        columns="experiment",
        values="accuracy_selected_strategy",
    )
    st.line_chart(accuracy_over_time, width="stretch")

    st.subheader("Experiment Summary Table")
    st.dataframe(summary_df, width="stretch")

    st.download_button(
        "Download experiment results as JSON",
        data=json.dumps(download_payload, indent=2),
        file_name="experiment_lab_results.json",
        mime="application/json",
    )
else:
    st.info("Choose settings and run the experiment matrix to compare multiple simulation scenarios.")
