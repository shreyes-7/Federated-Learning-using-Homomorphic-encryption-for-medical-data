# SecureFL: Federated Learning with Homomorphic Encryption for Medical Data

SecureFL is a privacy-preserving federated learning demo for medical data. It combines:

- Federated learning with local client training
- Homomorphic-encryption support through OpenFHE (`bgv` / `bfv`)
- Malicious client attack simulation
- Anomaly detection and robust aggregation
- Trust-aware aggregation, optional DP noise, and audit trail
- Streamlit dashboard with multipage analysis views

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
