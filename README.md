# Fraud-Spike / Transaction-Fraud Detector

[![Track](https://img.shields.io/badge/Track-02%20AI%20Risk%20Manager-blue.svg)](https://github.com/Parthrandive/AI-Risk-Manager)
[![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud%20Detection-orange.svg)](https://www.kaggle.com/c/ieee-fraud-detection)
[![Status](https://img.shields.io/badge/Status-In%20Development-green.svg)](https://github.com/Parthrandive/AI-Risk-Manager)

> **Razorpay AI Buildathon — Track 02: AI Risk Manager**  
> An end-to-end, leak-free, explainable transaction fraud detection engine with gray-zone triage abstention and honest evaluation metrics.

---

## 📌 Architecture & Pipeline Overview

```
Incoming transaction
        │
        ▼
[1] Data & Chronological Split
        │
        ▼
[2] Feature Engineering (Continuous State & Expanding Means)
        │
        ▼
[3] Gradient-Boosted Classifier (XGBoost / LightGBM)
        │
        ▼
[4] Evaluation & Honesty Layer
        │
        ▼
[5] SHAP Explainability + Gray-Zone Triage Gateway
        │
   ┌────┴───────────────┐
   ▼         ▼          ▼
Approve   Review      Block
(Low)   (Gray-Zone)   (High)
```

---

## 🏗️ Deep-Dive into Pipeline Layers

### 1. Data & Chronological Split
* **What it does**: Ingests `train_transaction.csv` and `train_identity.csv`, merges on `TransactionID`, runs data sanity checks, downcasts memory, sorts chronologically by `TransactionDT`, and performs a time-quantile split into train and test sets (never random shuffling).
* **Execution details**:
  - **Sanity Checks**: Drops exact duplicate rows; verifies `TransactionID` uniqueness; ensures non-negative `TransactionAmt`; checks `TransactionDT` monotonic safety; verifies `isFraud` is strictly binary $\{0, 1\}$.
  - **Memory Downcasting**: Converts `float64` $\to$ `float32` and `int64` $\to$ `int32`/`int16`/`int8` to ensure lightweight, fast tabular operations.
  - **Temporal Splitting**: Sorts by `TransactionDT` ascending; sets boundary at the 80th percentile; trains on $\le$ boundary, tests on $>$ boundary.
  - **Imbalance & Drift Reporting**: Evaluates and reports class imbalance (`isFraud.mean()`) and temporal drift between train and test splits.
  - **Zero Leakage Verification**: Guarantees 0% `TransactionID` overlap and strict temporal boundaries.

---

### 2. Feature Engineering & Preprocessing Engine
* **What it does**: Constructs a compact, interpretable set of behavioral, velocity, and recency features without lookahead leakage or boundary cold-start artifacts.
* **Core Identity Proxies**:
  - **Card Proxy**: Formed by combining `card1 + card2 + card3 + card4 + card5 + card6 + addr1 + addr2` to create an interpretable account/card fingerprint.
  - **Device Proxy**: Formed by combining `DeviceType + DeviceInfo + id_30 + id_31`.
    > *Note*: `DeviceInfo` is heavily missing (~70–80% null) in IEEE-CIS. When missing or first seen, `time_since_last_txn_device` defaults to `-1.0`, capturing anonymous/unlinked device behavior as an informative signal.
* **Leakage-Free Policies & Techniques**:
  - **Fit-on-Train Categorical Vocabulary**: Encodings (`ProductCD`, `card4`, `card6`, `P_emaildomain`, etc.) are learned from `train.parquet` only (`0: MISSING`, `1: UNSEEN` for novel test categories, `2+: Train categories`).
  - **Expanding Card Mean (Zero Intra-Split Leakage)**: `amt_to_expanding_card_mean_ratio` computes the ratio of the transaction amount against the *cumulative historical average of that card up to timestamp $t-1$*. Early transactions never reference future amounts from later in the dataset.
  - **Continuous State Across Boundary (Zero Cold-Start Artifacts)**: Velocity (10m, 1h, 24h) and recency state are computed in forward streaming succession across the train/test boundary. Transactions immediately after the split boundary retain full 24-hour historical context.

---

### 3. Gradient-Boosted Classifier
* **What it does**: Implements and trains tabular fraud classifiers with strict baseline comparisons.
* **Execution details**:
  - **Baseline First**: Implements baseline models (e.g., Logistic Regression / majority class) to establish an authentic precision/recall floor on the held-out test split.
  - **Primary Model**: Trains XGBoost / LightGBM gradient-boosted decision trees, compared transparently against baseline benchmarks.
* **Rationale**: Gradient-boosted trees provide high performance on tabular, imbalanced risk data, train fast without GPU locks, and expose direct feature importance metrics that feed Layer 5.

---

### 4. Evaluation & Honesty Layer
* **What it does**: Delivers transparent, real-world metric reporting without vanity dressing.
* **Evaluation Framework**:
  - **Precision & Recall**: Evaluated strictly on the held-out chronological test split (avoiding misleading accuracy or standalone ROC-AUC figures given the ~3–5% positive class imbalance).
  - **False-Positive Cost Analysis**: Explicitly evaluates flags raised per genuine fraud catch across varied decision thresholds.
  - **"What Didn't Work" Log**: A documented record of hypotheses, features, or modeling strategies that were tested, underperformed, and dropped with root-cause analysis.
* **Rationale**: Aligns directly with enterprise risk evaluation standards where business cost is driven by false positive friction and auditability.

---

### 5. SHAP Explainability + Gray-Zone Triage Gateway
* **What it does**: Converts continuous model risk scores into auditable, three-tier decisions with local feature attribution.
* **Triage Routing**:
  - 🟢 **Low Risk Score → Auto-Approve**: Seamless transaction flow.
  - 🟡 **Ambiguous Score (Middle Band) → Manual Review**: The model abstains gracefully on uncertain samples, sending them to the human triage queue.
  - 🔴 **High Risk Score → Auto-Block**: Real-time prevention of high-probability fraudulent activity.
* **Audit Trail**: Generates per-transaction SHAP explanations (e.g., *"Flagged due to 10-minute velocity spike + foreign merchant category"*).
* **Rationale**: Ensures resilience by avoiding false confidence in the gray zone, providing human-in-the-loop auditability.

---

## 🎯 Design Decisions & Scope

### Deliberately Excluded (Live Defense Focus)
Architectural options like GNN abuse-ring detectors (GraphSAGE), heavy graph storage, distributed caches (Redis/Aerospike), and ONNX compilation were evaluated and set aside. The focus of this implementation is end-to-end depth, rigor, leak-free validation, and explainability for live evaluation.

### Planned Future Work
- Graph-based identity clustering for syndicate fraud detection.
- Sub-millisecond latency optimizations with compiled inference engines.
- Adaptive thresholding based on dynamic merchant risk profiles.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git & Git LFS (if handling dataset archives)

### Installation
```bash
git clone git@github.com:Parthrandive/AI-Risk-Manager.git
cd AI-Risk-Manager
pip install -r requirements.txt
```

### Running the Pipeline

1. **Place Raw Data in `data/raw/`**:
   - `data/raw/train_transaction.csv`
   - `data/raw/train_identity.csv`

2. **Execute Layer 1 (Data & Chronological Split)**:
   ```bash
   python3 scripts/run_layer1.py
   ```

3. **Execute Layer 2 (Feature Engineering & Preprocessing)**:
   ```bash
   python3 scripts/run_layer2.py
   ```

4. **Run Verification Test Suite**:
   ```bash
   python3 -m pytest
   ```

---

## 👥 Authors & Acknowledgments
- **Team**: AI Risk Manager Team
- **Event**: Razorpay AI Buildathon (Track 02)
