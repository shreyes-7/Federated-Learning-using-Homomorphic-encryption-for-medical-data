# 🔐 Federated Learning using Homomorphic Encryption for Medical Data

A privacy-preserving machine learning framework combining:

- 🧠 Federated Learning (FL)
- 🔐 Homomorphic Encryption (HE) with OpenFHE (BGV/BFV)
- 🛡️ Malicious client attack simulation + anomaly detection
- 📊 Streamlit dashboard for explainability and demo readiness

This repository now includes both:

1. **Legacy notebook workflow** (original project assets), and
2. **Production-style modular Python pipeline** (clean, reusable, interview-ready).

---

## 📌 Why This Project Matters

In healthcare systems, patient data is highly sensitive and often cannot be centrally pooled due to compliance, policy, and trust constraints.

This project demonstrates a secure collaborative learning pattern where:

- Hospitals train locally on private data
- Only model updates (not raw records) leave each hospital
- Updates are encrypted before aggregation
- Server computes global updates over encrypted values
- Malicious participants can be detected and filtered

---

## 🧠 Core Theory

### Federated Learning (FL)

Federated Learning trains a shared global model across many clients (hospitals) without moving raw data.

At round `t`:

1. Server shares current global parameters `w_t`
2. Each client trains locally on private dataset shard
3. Client sends update `Δw_i = w_i - w_t`
4. Server aggregates updates to produce `w_(t+1)`

### Homomorphic Encryption (HE)

HE allows arithmetic directly on ciphertext.

In this project:

- Client updates are encrypted
- Server adds encrypted updates (no plaintext exposure)
- Aggregated ciphertext is decrypted to recover aggregated update

### Robustness Against Malicious Clients

We simulate Byzantine-like update poisoning and apply simple statistical filtering:

- Compute mean update
- Compute distance of each client update from mean
- Set threshold:

`threshold = mean(distance) + k * std(distance)`

- Flag clients above threshold
- Remove/neutralize them before final aggregation

This lets reviewers compare model quality:

- **With attack** (unfiltered)
- **After filtering** (defended)

---

## 🏗️ Architecture (Refactored)

```text
project/
├── core/
│   ├── fl_simulation.py      # End-to-end FL loop orchestration
│   ├── aggregation.py        # Aggregation helpers
│   └── model.py              # Logistic regression model (NumPy)
├── security/
│   ├── encryption.py         # Pluggable backends: simulated / openfhe
│   ├── attack.py             # Malicious update simulation
│   └── detection.py          # Distance-based anomaly detection
├── dashboard/
│   └── app.py                # Streamlit UI
├── utils/
│   └── helpers.py            # Dataset prep / splitting / reproducibility
├── data/
├── openfhe_lib/              # Existing C++ + Python wrappers (legacy-real HE path)
├── main.py                   # CLI runner
└── requirements.txt
```

---

## 🔄 End-to-End Functional Flow

1. Initialize global model and clients
2. Split dataset among clients
3. Local training on each client shard
4. Generate client updates
5. Optional malicious attack injection
6. Encrypt updates (`simulated` or `openfhe`)
7. Build baseline aggregation (with attack)
8. Run anomaly detection and flag suspicious clients
9. Filter/neutralize flagged updates
10. Aggregate filtered encrypted updates
11. Update global model
12. Track accuracy and diagnostics per round
13. Render results in Streamlit dashboard

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| FL Training | NumPy Logistic Regression |
| Dataset | scikit-learn breast cancer dataset |
| Encryption | OpenFHE wrappers (BGV/BFV) + simulation backend |
| Dashboard | Streamlit |
| Languages | Python + C++ |

---

## 🚀 Setup Guide

## 🟢 Option A: Quick Start (Refactored Pipeline)

Run from project root:

```bash
cd /home/nexushunter/SecureFL_crypto_project
```

Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run CLI (simulated encryption)

```bash
python main.py --backend simulated --num-clients 6 --rounds 8 --attack --attack-type scaling
```

### Run CLI (real OpenFHE path)

```bash
python main.py --backend openfhe --scheme bgv --num-clients 4 --rounds 3 --attack
```

### Run dashboard

```bash
streamlit run dashboard/app.py
```

Open: `http://localhost:8501`

---

## 🟢 Option B: Legacy/Original Environment Setup (Kept for Compatibility)

### 1) Setup WSL

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake build-essential libomp-dev wget
```

### 2) Install Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 3) Create Python Environment

```bash
conda create -n securefl python=3.10 -y
conda activate securefl
pip install --upgrade pip
pip install jupyter ipykernel numpy pandas matplotlib scikit-learn imbalanced-learn torch torchvision tqdm watermark
python -m ipykernel install --user --name securefl --display-name "Python (SecureFL)"
```

### 4) Install OpenFHE (v1.0.4 example)

```bash
cd ~
sudo rm -rf /usr/local/include/openfhe
sudo rm -rf /usr/local/lib/libOPENFHE*
sudo rm -rf openfhe-development
git clone --branch v1.0.4 https://github.com/openfheorg/openfhe-development.git
cd openfhe-development
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED=ON -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
make -j2
sudo make install
```

### Build scheme binaries in this repo

```bash
cd /home/nexushunter/SecureFL_crypto_project/openfhe_lib/bgv
rm -rf build && mkdir build && cd build
cmake ..
make -j2

cd ../../bfv
rm -rf build && mkdir build && cd build
cmake ..
make -j2

cd ../../ckks
rm -rf build && mkdir build && cd build
cmake ..
make -j2
```

---

## 🧪 Command Reference

### New Unique Modes (What You Added)

- Adaptive trust-weighted aggregation (`trust_weighted`)
- Robust alternatives (`trimmed_mean`, `coordinate_median`)
- Non-IID label-skew client partitioning
- Temporal feature drift simulation across rounds
- Optional differential privacy (DP) noise on client updates
- Verifiable audit hash chain for each round
- Automatic FedAvg fallback in OpenFHE mode when non-additive strategy is selected

### Core CLI arguments

```bash
python main.py \
  --backend simulated|openfhe \
  --scheme bgv|bfv \
  --aggregation-method fedavg|trimmed_mean|coordinate_median|trust_weighted \
  --trim-ratio 0.2 \
  --num-clients 4 \
  --rounds 8 \
  --local-epochs 2 \
  --lr 0.05 \
  --detection-k 1.0 \
  --seed 42 \
  --partition-mode iid|label_skew \
  --label-skew-strength 0.85 \
  --enable-drift \
  --drift-start-round 4 \
  --drift-strength 0.15 \
  --dp \
  --dp-noise-std 0.01 \
  --trust-beta 0.7 \
  --trust-min 0.1 \
  --trust-flag-penalty 0.5 \
  --disable-audit \
  --attack \
  --attack-type scaling|random \
  --malicious-fraction 0.25 \
  --scaling-factor 8.0 \
  --noise-std 5.0
```

### Examples

Simulated backend, random attack:

```bash
python main.py --backend simulated --num-clients 8 --rounds 10 --attack --attack-type random --noise-std 6.0
```

Unique stress test run (recommended demo):

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

OpenFHE backend (BGV):

```bash
python main.py --backend openfhe --scheme bgv --num-clients 4 --rounds 5 --attack
```

OpenFHE backend (BFV):

```bash
python main.py --backend openfhe --scheme bfv --num-clients 4 --rounds 5 --attack
```

---

## 📊 Dashboard Guide

Sidebar controls:

- Number of clients
- Rounds
- Local epochs
- Learning rate
- Detection threshold `k`
- Seed
- Encryption backend
- OpenFHE scheme
- Aggregation strategy
- Trust hyperparameters
- Encryption detail toggle
- Attack toggle and attack parameters
- Data mode (`iid` vs `label_skew`)
- Drift toggle and parameters
- DP toggle and DP-noise level
- Audit toggle
- FedAvg baseline comparison toggle

Main outputs:

- Final accuracy with attack
- Final accuracy after filtering
- Strategy comparison vs FedAvg baseline
- Accuracy over rounds chart
- Detection precision/recall over rounds
- Detection summary table
- Client update distance bars
- Trust evolution chart
- Audit chain table
- Encrypted update preview

---

## 🔁 Real HE vs Simulated HE

### Simulated backend

- Great for fast experimentation
- Flexible number of clients
- Uses masked-value toy encryption (not production crypto)
- Supports all advanced modes (trust, robust strategies, non-IID, drift, DP, audit)

### OpenFHE backend

- Uses existing OpenFHE wrappers and C++ binaries
- Demonstrates real HE encryption/aggregation path
- Current repository limitation: **fixed to 4 clients** in C++ server wrapper
- Non-additive strategies are auto-fallback to FedAvg in this mode

---

## 🔍 What Is Actually “Real” In This Project?

- Real local training and federated update flow: ✅
- Real attack injection and statistical anomaly detection: ✅
- Real OpenFHE execution path (in `openfhe` mode): ✅
- Live multi-hospital deployment integration (networked institutions, IAM, KMS/HSM, mTLS): ❌ (outside scope of demo)

This is an engineering-accurate prototype and evaluation harness for secure FL behavior.

---

## 🛡️ Security Notes and Limitations

- Includes robust strategies (`trimmed_mean`, `coordinate_median`, `trust_weighted`) and distance-based detection.
- Detection remains a lightweight heuristic; production should consider stronger Byzantine defenses (e.g., Krum/Bulyan family).
- OpenFHE wrapper currently uses file-based ciphertext exchange.
- Key management in this demo is local-file based, not HSM/KMS managed.
- Production deployment should include authenticated channels, audit logging, key rotation, and secure orchestration.

---

## 📚 Legacy Notebooks (Still Included)

These notebooks are retained for educational/reference value:

- `federated-learning-and-bgv-scheme.ipynb`
- `federated-learning-and-bfv-scheme.ipynb`
- `encrypted_learning.ipynb`

They are not the runtime entrypoint for the refactored modular pipeline.

---

## ▶️ Entry Points Summary

- CLI: `python main.py ...`
- Dashboard: `streamlit run dashboard/app.py`
- Legacy notebook mode: `jupyter notebook`

---

## 👨‍💻 Author

Shreyes  
Software Engineer  
GitHub: https://github.com/shreyes-7
