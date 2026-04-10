"""Attack Analysis page for detection behavior and anomaly severity."""

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

st.set_page_config(page_title="Attack Analysis", layout="wide")
st.title("Attack Analysis")
st.info(
    "Study how well the detector identifies malicious updates by comparing flagged "
    "clients, true malicious clients, update distances, and anomaly thresholds."
)


def _compute_detection_accuracy(detected: list[int], actual: list[int]) -> float:
    detected_set = set(detected)
    actual_set = set(actual)
    all_clients = detected_set | actual_set
    if not all_clients:
        return 1.0
    correct = len(detected_set & actual_set)
    missed = len(actual_set - detected_set)
    false_alarm = len(detected_set - actual_set)
    total = correct + missed + false_alarm
    return correct / total if total else 1.0


with st.sidebar:
    st.header("Simulation Settings")
    num_clients = st.slider("Number of clients", 2, 12, 6)
    num_rounds = st.slider("Rounds", 1, 25, 8)
    local_epochs = st.slider("Local epochs", 1, 10, 2)
    learning_rate = st.slider("Learning rate", 0.001, 0.2, 0.05)
    detection_k = st.slider("Detection threshold k", 0.5, 5.0, 1.0)
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)

    st.subheader("Aggregation Method")
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

    st.subheader("Attack Profile")
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

    run_clicked = st.button("Run Attack Analysis", type="primary")

if run_clicked:
    config = FLConfig(
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        detection_k=detection_k,
        seed=int(seed),
        encryption_backend="simulated",
        encryption_scheme="bgv",
        aggregation_method=aggregation_method,
        trim_ratio=trim_ratio,
    )
    attack_config = AttackConfig(
        enabled=True,
        attack_type=attack_type,
        malicious_fraction=malicious_fraction,
        scaling_factor=scaling_factor,
        random_noise_std=noise_std,
    )

    result = run_federated_simulation(config, attack_config)
    rounds_df = pd.DataFrame(result.rounds)

    rounds_df["detection_accuracy"] = rounds_df.apply(
        lambda row: _compute_detection_accuracy(
            row["detected_clients"],
            row["malicious_clients_ground_truth"],
        ),
        axis=1,
    )
    rounds_df["avg_update_distance"] = rounds_df["update_distances"].apply(
        lambda values: sum(values) / len(values) if values else 0.0
    )
    rounds_df["max_update_distance"] = rounds_df["update_distances"].apply(
        lambda values: max(values) if values else 0.0
    )
    rounds_df["attack_severity"] = rounds_df.apply(
        lambda row: (row["max_update_distance"] - row["threshold"]) / max(row["threshold"], 1e-8),
        axis=1,
    )

    st.subheader("Detected vs Actual Malicious Clients")
    st.dataframe(
        rounds_df[
            [
                "round_id",
                "detected_clients",
                "malicious_clients_ground_truth",
                "active_clients",
            ]
        ],
        width="stretch",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detection Metrics Per Round")
        st.line_chart(
            rounds_df.set_index("round_id")[["detection_precision", "detection_recall", "detection_accuracy"]],
            width="stretch",
        )
    with col2:
        st.subheader("Attack Severity")
        st.bar_chart(rounds_df.set_index("round_id")[["attack_severity"]], width="stretch")

    distance_threshold_df = rounds_df.set_index("round_id")[
        ["avg_update_distance", "max_update_distance", "threshold"]
    ]
    st.subheader("Update Distances vs Threshold")
    st.line_chart(distance_threshold_df, width="stretch")

    mean_detection_accuracy = float(rounds_df["detection_accuracy"].mean()) if not rounds_df.empty else 0.0
    mean_attack_severity = float(rounds_df["attack_severity"].mean()) if not rounds_df.empty else 0.0

    st.subheader("Insights")
    st.markdown(
        f"- Mean detection accuracy: `{mean_detection_accuracy:.3f}`\n"
        f"- Mean attack severity: `{mean_attack_severity:.3f}`\n"
        "- Positive severity means the strongest client update is above the anomaly threshold.\n"
        "- Higher detection precision means fewer false alarms; higher recall means more true malicious clients are caught."
    )

    st.download_button(
        "Download attack analysis as JSON",
        data=json.dumps(
            {
                "summary": result.summary,
                "rounds": rounds_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        file_name="attack_analysis.json",
        mime="application/json",
    )
else:
    st.info("Run a simulation with attacks enabled to inspect detection quality and anomaly severity.")
