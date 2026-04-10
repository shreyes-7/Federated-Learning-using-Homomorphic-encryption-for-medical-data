"""Trust Analysis page for client trust evolution and interpretation."""

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

st.set_page_config(page_title="Trust Analysis", layout="wide")
st.title("Trust Analysis")
st.info(
    "Inspect how client trust evolves across federated rounds, and how malicious "
    "behavior causes trust decay compared with benign participants."
)

with st.sidebar:
    st.header("Simulation Settings")
    num_clients = st.slider("Number of clients", 2, 12, 6)
    num_rounds = st.slider("Rounds", 1, 25, 8)
    local_epochs = st.slider("Local epochs", 1, 10, 2)
    learning_rate = st.slider("Learning rate", 0.001, 0.2, 0.05)
    detection_k = st.slider("Detection threshold k", 0.5, 5.0, 1.0)
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)

    st.subheader("Trust Settings")
    trust_beta = st.slider("Trust smoothing beta", 0.1, 0.95, 0.7)
    trust_min = st.slider("Minimum trust floor", 0.0, 0.5, 0.1)
    trust_penalty = st.slider("Flag penalty", 0.0, 1.0, 0.5)

    st.subheader("Attack Settings")
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

    run_clicked = st.button("Run Trust Analysis", type="primary")

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
        aggregation_method="trust_weighted",
        trust_beta=trust_beta,
        trust_min=trust_min,
        trust_flag_penalty=trust_penalty,
    )
    attack_config = AttackConfig(
        enabled=enable_attack,
        attack_type=attack_type,
        malicious_fraction=malicious_fraction,
        scaling_factor=scaling_factor,
        random_noise_std=noise_std,
    )

    result = run_federated_simulation(config, attack_config)
    rounds_df = pd.DataFrame(result.rounds)

    trust_long_rows: list[dict[str, object]] = []
    malicious_clients = set(rounds_df.iloc[0]["malicious_clients_ground_truth"]) if not rounds_df.empty else set()

    for _, row in rounds_df.iterrows():
        for client_id, trust_score in enumerate(row["trust_scores"]):
            trust_long_rows.append(
                {
                    "round_id": row["round_id"],
                    "client_id": client_id,
                    "trust_score": trust_score,
                    "client_type": "malicious" if client_id in malicious_clients else "benign",
                }
            )

    trust_long_df = pd.DataFrame(trust_long_rows)
    trust_wide_df = trust_long_df.pivot(index="round_id", columns="client_id", values="trust_score")
    trust_wide_df.columns = [f"client_{col}" for col in trust_wide_df.columns]

    st.subheader("Trust Evolution Per Client")
    st.line_chart(trust_wide_df, width="stretch")

    avg_trust_df = (
        trust_long_df.groupby(["round_id", "client_type"])["trust_score"]
        .mean()
        .reset_index()
        .pivot(index="round_id", columns="client_type", values="trust_score")
    )
    st.subheader("Average Trust: Benign vs Malicious")
    st.line_chart(avg_trust_df, width="stretch")

    final_trust_df = trust_long_df[trust_long_df["round_id"] == trust_long_df["round_id"].max()].copy()
    st.subheader("Final Round Trust Scores")
    st.dataframe(final_trust_df.sort_values(by="trust_score", ascending=False), width="stretch")

    benign_final = final_trust_df[final_trust_df["client_type"] == "benign"]["trust_score"]
    malicious_final = final_trust_df[final_trust_df["client_type"] == "malicious"]["trust_score"]

    benign_mean = float(benign_final.mean()) if not benign_final.empty else 0.0
    malicious_mean = float(malicious_final.mean()) if not malicious_final.empty else 0.0
    trust_gap = benign_mean - malicious_mean

    st.subheader("Insights")
    st.markdown(
        f"- Average final benign trust: `{benign_mean:.3f}`\n"
        f"- Average final malicious trust: `{malicious_mean:.3f}`\n"
        f"- Trust separation gap: `{trust_gap:.3f}`\n"
        "- A widening gap usually means the trust update rule is successfully isolating suspicious clients.\n"
        "- If trust stays similar for all clients, the attack may be weak or detection may be less sensitive."
    )

    st.download_button(
        "Download trust analysis data as JSON",
        data=json.dumps(
            {
                "summary": result.summary,
                "rounds": result.rounds,
                "trust_analysis": trust_long_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        file_name="trust_analysis.json",
        mime="application/json",
    )
else:
    st.info("Run a trust-weighted simulation to analyze how trust changes for each client over time.")
