what is """Streamlit dashboard for secure federated learning demo."""

from __future__ import annotations

import sys
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
st.title("Federated Learning + Homomorphic Encryption (Medical Demo)")

with st.sidebar:
    st.header("Simulation Controls")
    num_clients = st.slider("Number of clients", 2, 12, 4)
    num_rounds = st.slider("Rounds", 1, 25, 8)
    local_epochs = st.slider("Local epochs", 1, 10, 2)
    learning_rate = st.slider("Learning rate", 0.001, 0.2, 0.05)
    detection_k = st.slider("Detection threshold k", 0.5, 5.0, 1.0)
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)

    backend = st.selectbox("Encryption backend", ["simulated", "openfhe"])
    scheme = st.selectbox("OpenFHE scheme", ["bgv", "bfv"], disabled=(backend != "openfhe"))

    enable_encryption_view = st.toggle("Show encryption details", value=True)
    enable_attack = st.toggle("Enable attack simulation", value=True)

    attack_type = st.selectbox("Attack type", ["scaling", "random"], disabled=not enable_attack)
    malicious_fraction = st.slider("Malicious client fraction", 0.0, 0.9, 0.25, disabled=not enable_attack)
    scaling_factor = st.slider("Scaling attack factor", 1.0, 20.0, 8.0, disabled=(not enable_attack or attack_type != "scaling"))
    noise_std = st.slider("Random attack noise std", 0.1, 10.0, 5.0, disabled=(not enable_attack or attack_type != "random"))

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

        rounds_df = pd.DataFrame(result.rounds)

        col1, col2 = st.columns(2)
        col1.metric("Final Accuracy (With Attack)", f"{result.final_accuracy_with_attack:.4f}")
        col2.metric("Final Accuracy (After Filtering)", f"{result.final_accuracy_after_filtering:.4f}")

        st.subheader("Accuracy Over Rounds")
        st.line_chart(
            rounds_df[["accuracy_with_attack", "accuracy_after_filtering"]],
            width="stretch",
        )

        st.subheader("Detection Summary")
        detect_df = rounds_df[["round_id", "detected_clients", "malicious_clients_ground_truth", "threshold"]]
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

        if enable_encryption_view:
            st.subheader("Encrypted Update Preview")
            round_choice = st.slider("Round to inspect", 1, len(rounds_df), 1)
            selected = rounds_df.iloc[round_choice - 1]
            preview_df = pd.DataFrame(
                {
                    "client_id": list(range(len(selected["encrypted_preview"]))),
                    "ciphertext_preview": selected["encrypted_preview"],
                    "update_norm": selected["client_update_norms"],
                }
            )
            st.dataframe(preview_df, width="stretch")

        st.success("Simulation completed.")
else:
    st.info("Set controls and click 'Run Simulation' to start.")
