"""Strategy Comparison page for aggregation-method benchmarking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fl_simulation import FLConfig, run_federated_simulation
from security.attack import AttackConfig

st.set_page_config(page_title="Strategy Comparison", layout="wide")
st.title("Strategy Comparison")
st.info(
    "Benchmark all supported aggregation strategies on the same simulation setup to "
    "see which one balances accuracy and malicious-client resilience best."
)

STRATEGIES = ["fedavg", "trust_weighted", "trimmed_mean", "coordinate_median"]

with st.sidebar:
    st.header("Simulation Settings")
    num_clients = st.slider("Number of clients", 2, 12, 6)
    num_rounds = st.slider("Rounds", 1, 25, 8)
    local_epochs = st.slider("Local epochs", 1, 10, 2)
    learning_rate = st.slider("Learning rate", 0.001, 0.2, 0.05)
    detection_k = st.slider("Detection threshold k", 0.5, 5.0, 1.0)
    trim_ratio = st.slider("Trim ratio", 0.0, 0.45, 0.2)
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)

    st.subheader("Attack Profile")
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

    run_clicked = st.button("Compare Strategies", type="primary")

if run_clicked:
    attack_config = AttackConfig(
        enabled=enable_attack,
        attack_type=attack_type,
        malicious_fraction=malicious_fraction,
        scaling_factor=scaling_factor,
        random_noise_std=noise_std,
    )

    rows: list[dict[str, object]] = []
    rounds_frames: list[pd.DataFrame] = []

    progress = st.progress(0.0, text="Starting strategy comparison...")
    for idx, strategy in enumerate(STRATEGIES, start=1):
        config = FLConfig(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            detection_k=detection_k,
            seed=int(seed),
            encryption_backend="simulated",
            encryption_scheme="bgv",
            aggregation_method=strategy,
            trim_ratio=trim_ratio,
            partition_mode=partition_mode,
            label_skew_strength=label_skew_strength,
        )

        progress.progress(idx / len(STRATEGIES), text=f"Running {strategy}")
        result = run_federated_simulation(config, attack_config)
        rounds_df = pd.DataFrame(result.rounds)
        rounds_df["strategy"] = strategy
        rounds_frames.append(rounds_df)

        rows.append(
            {
                "strategy": strategy,
                "final_accuracy": result.final_accuracy_selected_strategy,
                "attacked_accuracy": result.final_accuracy_with_attack,
                "mean_detection_precision": result.summary.get("mean_detection_precision", 0.0),
                "mean_detection_recall": result.summary.get("mean_detection_recall", 0.0),
                "final_mean_trust_benign": result.summary.get("final_mean_trust_benign", 0.0),
                "final_mean_trust_malicious": result.summary.get("final_mean_trust_malicious", 0.0),
                "audit_chain_valid": result.audit_chain_valid,
            }
        )

    progress.empty()

    summary_df = pd.DataFrame(rows).sort_values(by="final_accuracy", ascending=False)
    rounds_all_df = pd.concat(rounds_frames, ignore_index=True)

    best_strategy = summary_df.iloc[0]["strategy"]
    st.success(f"Best performing strategy in this run: {best_strategy}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Accuracy")
        st.bar_chart(summary_df.set_index("strategy")[["final_accuracy", "attacked_accuracy"]], width="stretch")
    with col2:
        st.subheader("Detection Performance")
        st.bar_chart(
            summary_df.set_index("strategy")[["mean_detection_precision", "mean_detection_recall"]],
            width="stretch",
        )

    st.subheader("Accuracy Over Rounds")
    accuracy_rounds = rounds_all_df.pivot(index="round_id", columns="strategy", values="accuracy_selected_strategy")
    st.line_chart(accuracy_rounds, width="stretch")

    st.subheader("Strategy Summary")
    st.dataframe(summary_df, width="stretch")

    st.markdown(
        "- `fedavg` is the simplest baseline.\n"
        "- `trust_weighted` uses evolving trust scores.\n"
        "- `trimmed_mean` drops extreme coordinates before averaging.\n"
        "- `coordinate_median` is more resistant to outlier updates at each parameter coordinate."
    )

    st.download_button(
        "Download strategy comparison as JSON",
        data=json.dumps(summary_df.to_dict(orient="records"), indent=2),
        file_name="strategy_comparison.json",
        mime="application/json",
    )
else:
    st.info("Run the page to compare all aggregation strategies on a common setup.")
