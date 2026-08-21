# Fraud-Spike / Transaction-Fraud Detector

[![Track](https://img.shields.io/badge/Track-02%20AI%20Risk%20Manager-blue.svg)](https://github.com/Parthrandive/AI-Risk-Manager)
[![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud%20Detection%20(590k%20txns)-orange.svg)](https://www.kaggle.com/c/ieee-fraud-detection)
[![Status](https://img.shields.io/badge/Status-Validated%20on%20Real%20Data-green.svg)](https://github.com/Parthrandive/AI-Risk-Manager)

> **Razorpay AI Buildathon — Track 02: AI Risk Manager**  
> An end-to-end, leak-free, explainable transaction fraud detection engine with gray-zone triage abstention and honest evaluation metrics.

---

## 📌 Architecture & Pipeline Overview

```
Incoming transaction
        │
        ▼
[1] Data & Chronological Split (590k IEEE-CIS, 80/20 Time-Quantile)
        │
        ▼
[2] Feature Engineering (Continuous Streaming State & Expanding Means)
        │
        ▼
[3] Gradient-Boosted Classifiers (XGBoost & LightGBM vs. Logistic Baseline)
        │
        ▼
[4] Evaluation & Honesty Layer (Multi-Threshold Sweeps & Cost Curves)
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

## 🏆 Real IEEE-CIS Dataset Benchmark Results

Evaluated strictly on the held-out chronological test split (**118,108 transactions**, 4,064 true frauds, **3.441% test fraud rate**):

### 1. Model Performance vs. Baseline

| Model | PR-AUC (Primary Metric) | ROC-AUC | Precision (at 0.5) | Recall (at 0.5) | Brier Score Loss (Calibration) |
|---|---|---|---|---|---|
| **Baseline (Logistic Regression)** | `0.1834` | `0.8310` | `12.67%` | `69.24%` | `0.0682` |
| **LightGBM Classifier** | `0.4900` *(+0.3065)* | `0.8901` | `81.64%` | `30.95%` | `0.0238` |
| **XGBoost GBDT (Primary)** | **`0.5149`** *(+0.3315)* | **`0.8975`** | **`81.08%`** | **`30.78%`** | **`0.0224`** |

---

### 2. Multi-Threshold Decision Sweep & Synthesized Operational Policies

Rather than cherry-picking an uncalibrated default 0.5 cutoff, the pipeline sweeps decision boundaries from $0.001 \to 0.950$ (confirming the net financial benefit peak at $\tau = 0.010$) and synthesizes the spectrum into **3 named shipping policies**:

```
Decision Threshold Sweep Spectrum:
0.001 ──→ 0.010 [Aggressive] ──→ 0.100 ──→ 0.300 [Balanced] ──→ 0.500 ──→ 0.800 [Conservative] ──→ 0.950
```

| Policy Profile | Operating Threshold ($\tau$) | Recall (Catch Rate) | Precision | False Positives per True Catch ($\text{FP}/\text{TP}$) | Dollar Capture Rate | Auto-Approved Volume | Strategic Intent |
|---|---|---|---|---|---|---|---|
| 🟡 **Aggressive Policy** *(Max Catch & Peak Net Benefit)* | **`0.010`** | **`94.78%`** | `6.30%` | `14.87` | **`94.57%`** *(+\$529.1k Net Benefit)* | `48.25%` | Global maximum on net chargeback recovery curve; captures 95% of all fraud dollars, absorbing higher analyst triage load. |
| 🟢 **Balanced Policy** *(Optimal F1 / Low Friction)* | **`0.300`** | **`38.83%`** | **`67.81%`** | **`0.47`** *(>2 true catches / 1 FP)* | `39.88%` | **`98.03%`** | Standard production baseline maximizing F1 (`0.4938`); flags only 1.97% of traffic with <0.5 false alerts per catch. |
| 🔴 **Conservative Policy** *(Minimal Customer Friction)* | **`0.800`** | **`20.20%`** | **`90.82%`** | **`0.10`** *(10 true catches / 1 FP)* | `21.40%` | **`99.23%`** | Ultra-low customer friction; 90.8% precision with near-zero false alarms for frictionless VIP checkout conversion. |

> [!NOTE]
> **Stated Operational Cost Assumptions**:
> - **Manual Analyst Review Cost**: Assumed at **\$5.00** per flagged case (industry benchmark for 3–5 min analyst triage).
> - **Chargeback Loss Multiplier**: Assumed at **1.5x** the transaction dollar amount (goods loss + merchant penalty fees).

---

## 🏗️ Deep-Dive into Pipeline Layers

### 1. Data & Chronological Split
* **Execution**: Ingests `590,540` rows (434 columns), downcasts memory from `1.98 GB` $\to$ `1.07 GB` (-45.9%), sorts strictly by `TransactionDT` ascending, and splits at the 80th percentile boundary (`12,192,854s`).
* **Train Split**: `472,432` records (16,599 frauds, `3.514%` rate).
* **Test Split**: `118,108` records (4,064 frauds, `3.441%` rate).
* **Leakage Verification**: 100% Passed (0 overlapping `TransactionID`s, strict non-overlapping temporal cut).

---

### 2. Feature Engineering & Preprocessing Engine
* **Identity Proxies**:
  - **Card Proxy**: Synthesized from `card1 + card2 + card3 + card4 + card5 + card6 + addr1 + addr2`.
  - **Device Proxy**: Synthesized from `DeviceType + DeviceInfo + id_30 + id_31` *(Note: `DeviceInfo` is ~70% null in IEEE-CIS; unlinked recency cleanly defaults to `-1.0` as an informative signal)*.
* **Leakage Safeguards**:
  - **Expanding Cumulative Card Mean**: `amt_to_expanding_card_mean_ratio` computes the ratio against the cumulative average up to $t-1$, eliminating intra-train lookahead leakage.
  - **Continuous Streaming State**: `StreamingRiskState` seamlessly carries 10m/1h/24h rolling velocity counters across the train/test boundary, eliminating test cold-start artifacts.
* **Feature Set**: `427` model-ready numeric and label-encoded features.

---

### 3. Gradient-Boosted Classifiers (XGBoost & LightGBM)
* **Execution**: Trains `XGBoost` and `LightGBM` GBDTs alongside a `LogisticRegression` baseline.
* **Feature Importances (Top Gain)**:
  1. `V257` *(Gain: 3,961.83)*
  2. `V258` *(Gain: 2,536.33)*
  3. `V201` *(Gain: 982.93)*
  4. `id_11` *(Gain: 615.70)*
  5. `C4` *(Gain: 599.07)*

---

### 4. Evaluation & Honesty Layer
* Implements multi-threshold sweeping, calibration analysis (Brier score loss: `0.0224`), dollar capture curves, and the structured "What Didn't Work" registry.

---

### 5. SHAP Explainability + Gray-Zone Triage Gateway
* **Triage Routing**:
  - 🟢 **Low Risk ($\le \tau_{\text{low}}$) → Auto-Approve**: Seamless zero-friction checkout.
  - 🟡 **Gray-Zone ($\tau_{\text{low}} < \text{score} < \tau_{\text{high}}$) → Manual Review**: The model abstains gracefully on uncertain samples.
  - 🔴 **High Risk ($\ge \tau_{\text{high}}$) → Auto-Block**: Real-time prevention of high-probability fraudulent activity.
* **Audit Trail**: Plain-language local SHAP feature contribution breakdown attached to every flagged case.

---

## 📝 "What Didn't Work" Registry & Lessons Learned

| Attempted Approach | Outcome | Root Cause Analysis & Empirical Pivot |
|---|---|---|
| **Random K-Fold / Shuffling** | *Rejected* | Random splits leak future card velocities and identity fingerprints into past predictions. Replaced with strict 80th-percentile time split. |
| **Static Per-Card Historical Mean** | *Refactored* | Static train-wide averages allowed $t=100$ transactions to reference spend behavior from $t=5000$. Refactored to expanding cumulative mean up to $t-1$. |
| **Independent Per-Split Rolling Windows** | *Refactored* | Resetting velocity windows at the boundary caused early test transactions to register as "first-ever" events. Refactored to streaming state continuity. |
| **Hardcoded `scale_pos_weight = 28.87`** | *Parameterized* | Multiplying positive gradients by ~29 forced trees into noisy leaf splits, degrading PR-AUC from `0.5149` to `0.4629` and Brier score from `0.0224` to `0.1118`. Parameterized as a sweep. |
| **Raw 300+ Anonymized V-Features Alone** | *De-prioritized* | Opaque features lack explainability for human risk ops. Augmented with interpretable velocity, recency, and expanding ratios for Layer 5 auditability. |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git & Git LFS

### Installation
```bash
git clone git@github.com:Parthrandive/AI-Risk-Manager.git
cd AI-Risk-Manager
pip install -r requirements.txt
```

### Running the End-to-End Pipeline

1. **Place IEEE-CIS CSVs in `data/raw/`**:
   - `data/raw/train_transaction.csv`
   - `data/raw/train_identity.csv`

2. **Execute Layers 1 to 4**:
   ```bash
   python3 scripts/run_layer1.py
   python3 scripts/run_layer2.py
   python3 scripts/run_layer3.py
   python3 scripts/run_layer4.py
   ```

3. **Run Automated Test Suite**:
   ```bash
   python3 -m pytest
   ```

---

## 👥 Authors & Acknowledgments
- **Team**: AI Risk Manager Team
- **Event**: Razorpay AI Buildathon (Track 02)
