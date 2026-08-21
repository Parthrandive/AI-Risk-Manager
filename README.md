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
[2] Feature Engineering
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
* **What it does**: Ingests `train_transaction.csv` and `train_identity.csv`, merges on `TransactionID`, sorts chronologically by `TransactionDT`, and performs a time-quantile split into train and test sets (never random shuffling).
* **Execution details**:
  - Sorts all rows by `TransactionDT` in ascending order.
  - Places a split boundary at a selected percentile (default: 80th percentile).
  - Assigns rows on or before the boundary to training; rows after to test.
  - Evaluates and reports fraud rate (class imbalance) independently for train and test sets.
  - Leakage verification: guarantees 0% `TransactionID` overlap and strict temporal boundaries (no training rows past the boundary, no test rows before it).
* **Rationale**: Eliminates lookahead bias where future data informs past predictions—ensuring the model reflects a live production checkout flow.

---

### 2. Feature Engineering
* **What it does**: Constructs a compact, interpretable set of behavioral and transactional features rather than relying blindly on the raw 300+ anonymized `V1`–`V339` columns.
* **Key Features**:
  - Transaction amount & relative distribution.
  - Recency: Time-since-last-transaction per card/device.
  - Merchant Category Code (MCC) interactions.
  - Rolling Velocity Counts: Transaction frequency across short and medium temporal windows (e.g., last 10 minutes, last 24 hours per card/device).
* **Rationale**: All velocity and rolling features are computed in strict `TransactionDT` chronological order over time-split data without resetting per-slice or referencing future states. High interpretability ensures human operators and evaluators can validate risk rationale.

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
```

### Dataset Setup
Download the IEEE-CIS Fraud Detection dataset files into `data/`:
- `train_transaction.csv`
- `train_identity.csv`

---

## 👥 Authors & Acknowledgments
- **Team**: AI Risk Manager Team
- **Event**: Razorpay AI Buildathon (Track 02)
