# 🔐 Federated Learning using Homomorphic Encryption for Medical Data

A privacy-preserving machine learning framework combining:

-   🧠 Federated Learning (FL)
-   🔐 Homomorphic Encryption (OpenFHE)
-   🏥 Multi-hospital collaborative training
-   🔒 BGV & BFV encryption schemes

This project demonstrates secure collaborative learning where multiple
hospitals train a shared model **without sharing raw patient data**.

------------------------------------------------------------------------

# 📌 Project Overview

In healthcare systems:

-   Patient data is highly sensitive.
-   Regulations prevent raw data sharing.
-   Centralized machine learning risks privacy violations.

This project solves the problem using:

Federated Learning + Homomorphic Encryption

✔ Each hospital trains locally\
✔ Model weights are encrypted\
✔ Server aggregates encrypted weights\
✔ No raw data is shared

------------------------------------------------------------------------

# 🏗️ System Architecture

Windows\
↓\
WSL2 (Linux)\
↓\
OpenFHE v1.0.4\
↓\
SecureFL C++ (BGV/BFV Encryption)\
↓\
Conda Python 3.10\
↓\
Jupyter Notebook\
↓\
Federated Learning + Encrypted Aggregation

------------------------------------------------------------------------

# ⚙️ Technologies Used

  Component            Technology
  -------------------- ------------------------
  ML Framework         PyTorch
  Encryption Library   OpenFHE v1.0.4
  Encryption Schemes   BGV & BFV
  Languages            Python + C++
  Environment          Conda
  IDE                  VS Code (Remote - WSL)

------------------------------------------------------------------------

# 🚀 Setup Guide

## 🟢 1. Setup WSL

``` bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake build-essential libomp-dev wget
```

------------------------------------------------------------------------

## 🟢 2. Install Miniconda

``` bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Restart terminal and verify:

``` bash
conda --version
```

------------------------------------------------------------------------

## 🟢 3. Create Python Environment

``` bash
conda create -n securefl python=3.10 -y
conda activate securefl
```

Install dependencies:

``` bash
pip install --upgrade pip
pip install jupyter ipykernel numpy pandas matplotlib scikit-learn imbalanced-learn torch torchvision tqdm
```

Register kernel:

``` bash
python -m ipykernel install --user --name securefl --display-name "Python (SecureFL)"
```

------------------------------------------------------------------------

## 🟢 4. Install OpenFHE (v1.0.4 Required)

Remove old versions:

``` bash
cd ~
sudo rm -rf /usr/local/include/openfhe
sudo rm -rf /usr/local/lib/libOPENFHE*
sudo rm -rf openfhe-development
```

Clone correct version:

``` bash
git clone --branch v1.0.4 https://github.com/openfheorg/openfhe-development.git
cd openfhe-development
```

Build:

``` bash
mkdir build
cd build

cmake ..  -DCMAKE_INSTALL_PREFIX=/usr/local  -DBUILD_SHARED=ON  -DBUILD_UNITTESTS=OFF  -DBUILD_EXAMPLES=OFF  -DBUILD_BENCHMARKS=OFF

make -j2
sudo make install
```

Verify:

``` bash
find /usr/local/include -name binfhecontext.h
```

Expected output:

    /usr/local/include/openfhe/binfhecontext.h

------------------------------------------------------------------------

## 🟢 5. Clone This Repository

``` bash
git clone https://github.com/shreyes-7/Federated-Learning-using-Homomorphic-encryption-for-medical-data.git
cd Federated-Learning-using-Homomorphic-encryption-for-medical-data
```

------------------------------------------------------------------------

## 🟢 6. Build BGV Scheme

``` bash
cd openfhe_lib/bgv
rm -rf build
mkdir build
cd build
cmake ..
make -j2
```

------------------------------------------------------------------------

## 🟢 7. Build BFV Scheme

``` bash
cd ../../bfv
rm -rf build
mkdir build
cd build
cmake ..
make -j2
```

------------------------------------------------------------------------

## 🟢 8. Run Notebook

``` bash
conda activate securefl
jupyter notebook
```

Open:

federated-learning-and-bgv-scheme.ipynb

Select kernel:

Python (SecureFL)

Restart Kernel → Run All.

------------------------------------------------------------------------

# 🔐 Encryption Workflow

1.  Each hospital trains locally.
2.  Local model weights are encrypted.
3.  Encrypted weights are sent to server.
4.  Server performs homomorphic aggregation.
5.  Aggregated ciphertext is sent back.
6.  Clients decrypt global model.

No raw patient data is shared at any stage.

------------------------------------------------------------------------

# 📊 Federated Learning Flow

Client → Local Training → Encrypt Weights → Server Aggregation → Decrypt
→ Continue Training

------------------------------------------------------------------------

# 🏥 Use Case

-   Multi-hospital disease prediction\
-   Privacy-preserving medical AI\
-   Secure collaborative research

------------------------------------------------------------------------

# 👨‍💻 Author

Shreyes\
Software Engineer

GitHub: https://github.com/shreyes-7

------------------------------------------------------------------------

# 📌 Future Improvements

-   Docker containerization\
-   Performance benchmarking\
-   Differential privacy integration\
-   Production-ready deployment
