# SecureFL: Federated Learning with Homomorphic Encryption for Medical Data

SecureFL is a privacy-preserving federated learning demo for medical data. It combines:

- Federated learning with local client training
- Homomorphic-encryption support through OpenFHE (`bgv` / `bfv`)
- Malicious client attack simulation
- Anomaly detection and robust aggregation
- Trust-aware aggregation, optional DP noise, and audit trail
- Streamlit dashboard with multipage analysis views

## Project Overview

This project demonstrates how multiple medical institutions can collaboratively train a machine-learning model without sharing raw patient data with a central server.

In a traditional centralized workflow, all data is collected in one place before training. In many healthcare settings, that is difficult or undesirable because patient records are sensitive, regulated, and often distributed across different institutions. SecureFL models a safer alternative:

- each client trains locally on its own data shard
- only model updates are sent to the server
- those updates can be protected with encryption
- suspicious or malicious client behavior can be detected and reduced during aggregation

The result is a learning system that is not only privacy-aware, but also built to study robustness under adversarial conditions.

## What The Project Actually Simulates

At a high level, the system runs a full federated-learning round loop:

1. A global logistic-regression model is initialized.
2. The training dataset is split across several simulated clients.
3. Each client trains locally for a few epochs.
4. The difference between the local model and global model becomes that client's update.
5. Optional malicious behavior is injected into some client updates.
6. Optional differential privacy noise can be added.
7. Updates are encrypted using either:
   - a lightweight simulated backend, or
   - the OpenFHE-backed path for `bgv` and `bfv`
8. The server analyzes client-update behavior to detect anomalies.
9. Aggregation is performed using the selected strategy.
10. Trust scores and audit information are updated round by round.
11. Accuracy and security-related metrics are collected for visualization and comparison.

This makes the repository useful both as:

- a demo of secure federated learning concepts
- an experimentation tool for testing aggregation strategies and attack resilience

## Core Features Explained

### Federated Learning

The main learning pipeline is a simple but clear federated-learning simulation. Instead of moving data to the server, the server sends the current model to clients, each client trains locally, and only parameter updates are returned.

### Homomorphic Encryption

The project supports an encrypted aggregation path using OpenFHE. This allows the server to operate on encrypted model updates instead of reading plaintext updates directly. A simulated backend is also included so the full pipeline can be tested quickly without requiring OpenFHE installation.

### Attack Simulation

The code can simulate malicious clients that try to poison training. Two attack types are currently supported:

- `scaling`: malicious clients multiply their updates to make them dominate aggregation
- `random`: malicious clients inject noisy updates to disrupt training

### Detection And Robust Aggregation

The project measures how far each client update is from the average behavior and flags suspicious updates. It also supports multiple aggregation strategies so you can compare how the system behaves under attack:

- `fedavg`
- `trimmed_mean`
- `coordinate_median`
- `trust_weighted`

### Trust Scoring

Each client receives a dynamic trust score over rounds. Clients that behave consistently remain more trusted, while clients flagged as anomalous are penalized. This trust score is used directly by the `trust_weighted` aggregation strategy.

### Differential Privacy And Auditability

The simulator can optionally add DP-style Gaussian noise to client updates. It also supports a round-wise audit hash chain so each round transition can be logged in a tamper-evident way for inspection in the dashboard.

## Execution Flow

The main execution path is:

- `main.py` or `dashboard/app.py` builds `FLConfig` and `AttackConfig`
- `core/fl_simulation.py` runs the end-to-end simulation
- `utils/helpers.py` prepares and partitions the dataset
- `core/model.py` trains the NumPy logistic-regression model on each client
- `security/attack.py` optionally modifies malicious client updates
- `security/encryption.py` encrypts and aggregates client updates
- `security/detection.py` identifies suspicious updates
- `security/trust.py` updates trust scores
- `security/audit.py` creates and verifies the audit chain
- the CLI prints summaries, while Streamlit renders charts, tables, and comparisons

## Why The Dashboard Matters

The dashboard is not just a visualization layer. It turns the project into an analytical tool by making it easier to compare behaviors under different settings.

The main dashboard shows:

- final accuracy with and without filtering
- detection precision and recall
- client update distances
- trust evolution
- audit-chain preview
- encrypted update previews

The multipage analytics views extend that further with:

- experiment matrix comparisons
- strategy benchmarking
- trust-specific analysis
- attack-specific analysis

## Project Structure

```text
SecureFL_crypto_project/
├── main.py              # CLI entry point
├── core/                # FL simulation, model, aggregation
├── security/            # encryption, attacks, detection, trust, audit
├── utils/               # dataset prep and helpers
├── dashboard/
│   ├── app.py           # main Streamlit dashboard
│   └── pages/           # extra Streamlit analytical pages
├── openfhe_lib/         # OpenFHE wrappers and C++ binaries
├── data/                # datasets and OpenFHE key/ciphertext files
└── requirements.txt
```

## Clone And Setup

### 1. Clone the repository

```bash
git clone https://github.com/shreyes-7/Federated-Learning-using-Homomorphic-encryption-for-medical-data.git
cd Federated-Learning-using-Homomorphic-encryption-for-medical-data
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Project

### Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

### Run the CLI simulation

```bash
python main.py --backend simulated --num-clients 6 --rounds 8 --attack --attack-type scaling
```

## Simple Run Commands

### Simulated backend

This works immediately after `pip install -r requirements.txt`.

```bash
python main.py --backend simulated --num-clients 6 --rounds 8 --attack --attack-type scaling
```

### OpenFHE backend

This requires OpenFHE installed and the OpenFHE binaries in this repo built first.

```bash
python main.py --backend openfhe --scheme bgv --num-clients 4 --rounds 3 --attack
```

You can also run BFV:

```bash
python main.py --backend openfhe --scheme bfv --num-clients 4 --rounds 3 --attack
```

## OpenFHE Setup

Install system packages:

```bash
sudo apt update
sudo apt install -y git cmake build-essential libomp-dev wget
```

Install OpenFHE:

```bash
cd ~
git clone --branch v1.0.4 https://github.com/openfheorg/openfhe-development.git
cd openfhe-development
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED=ON -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
make -j2
sudo make install
```

Build the repo OpenFHE wrappers:

### BGV

```bash
cd openfhe_lib/bgv
rm -rf build
mkdir build
cd build
cmake ..
make -j2
```

### BFV

```bash
cd openfhe_lib/bfv
rm -rf build
mkdir build
cd build
cmake ..
make -j2
```

## Dashboard Pages

When you run:

```bash
streamlit run dashboard/app.py
```

you can access:

- Main dashboard
- Experiment Lab
- Strategy Comparison
- Trust Analysis
- Attack Analysis

## Useful Notes

- `simulated` backend is the fastest way to test the project.
- `openfhe` mode currently expects exactly `4` clients.
- If `openfhe` is selected with a non-additive strategy, the code may fall back to `fedavg`.
- The Streamlit multipage files live in `dashboard/pages/`.

## Main CLI Options

```bash
python main.py \
  --backend simulated|openfhe \
  --scheme bgv|bfv \
  --num-clients 4 \
  --rounds 8 \
  --local-epochs 2 \
  --lr 0.05 \
  --aggregation-method fedavg|trimmed_mean|coordinate_median|trust_weighted \
  --attack \
  --attack-type scaling|random
```

## Recommended Quick Demo

```bash
python main.py \
  --backend simulated \
  --num-clients 8 \
  --rounds 8 \
  --aggregation-method trust_weighted \
  --partition-mode label_skew \
  --enable-drift \
  --dp \
  --attack \
  --attack-type scaling
```
