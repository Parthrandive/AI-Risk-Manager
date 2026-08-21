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

Rather than cherry-picking an uncalibrated default 0.5 cutoff, the pipeline sweeps decision boundaries from $0.001 \to 0.950$ and synthesizes them into actionable enterprise shipping policies:

```
Decision Threshold Sweep Spectrum:
0.001 ──→ 0.010 [Theoretical Limit] ──→ 0.100 ──→ 0.200 [Primary Ship] ──→ 0.300 [Balanced F1] ──→ 0.800 [VIP] ──→ 0.950
```

| Policy Profile | Operating Threshold ($\tau$) | Catch Rate (Recall) | Precision | False Positives per True Catch ($\text{FP}/\text{TP}$) | Flagged Volume | Auto-Approved Volume | Strategic Intent & Operational Context |
|---|---|---|---|---|---|---|---|
| 🌟 **Production Policy** *(Capacity-Constrained)* | **`0.200`** | **`44.86%`** *(1,823 caught)* | **`54.89%`** | **`0.82`** *(>1.2 true catches / 1 FP)* | **`2.81%`** *(3,321 txns)* | **`97.19%`** | **RECOMMENDED PRIMARY SHIP CANDIDATE.** Designed for fixed enterprise risk teams with a $\le 3.0\%$ review headcount budget. Captures 45% of fraud with more true catches than false alarms. |
| 🟢 **Balanced Policy** *(Optimal F1 / Queue Defense)* | **`0.300`** | **`38.83%`** | **`67.81%`** | **`0.47`** *(>2 true catches / 1 FP)* | `1.97%` | **`98.03%`** | Standard production baseline maximizing harmonic F1 (`0.4938`); flags <2% of traffic. *Optimizes classification balance and queue protection, accepting higher missed chargebacks than the Primary Ship policy.* |
| 🔴 **Conservative Policy** *(Minimal Friction VIP)* | **`0.800`** | **`20.20%`** | **`90.82%`** | **`0.10`** *(10 true catches / 1 FP)* | `0.77%` | **`99.23%`** | Ultra-low customer friction; 90.8% precision with near-zero false alarms for frictionless VIP checkout conversion. |
| ⚠️ **Theoretical Ceiling** *(Unconstrained Math Limit)* | **`0.010`** | **`94.78%`** | `6.30%` | `14.87` | `51.75%` *(61k reviews)* | `48.25%` | **THEORETICAL UPPER BOUND (NOT SHIPPED).** Mathematical peak of unconstrained formula (+\$529.1k). Unviable for live deployment without massive reviewer headcount. |

> [!NOTE]
> **Stated Operational Cost Assumptions & Sensitivity**:
> - **Manual Review Cost**: Assumed at **\$5.00** per flagged case (industry benchmark for 3–5 min analyst triage). Sensitivity sweep (\$2 $\to$ \$15) confirms stability: under capacity constraints, the operating point remains firmly anchored at $\tau \approx 0.19 - 0.20$.
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
### 4. Evaluation & Honesty Layer
* Implements multi-threshold sweeping ($0.001 \to 0.950$), probability calibration verification (Brier score: `0.0224`), dollar capture curves, cost sensitivity sweeps (\$2 $\to$ \$15), and the structured "What Didn't Work" registry.

---

### 5. SHAP Explainability + Gray-Zone Triage Gateway
* **Grounded 3-Way Triage Gateway**:
  - 🟢 **Auto-Approve (`score < 0.150`)**: **`113,756` transactions (`96.32%` of volume)** approved instantly with zero customer friction and zero reviewer delay.
  - 🟡 **Manual Review / Gray-Zone (`0.150 <= score < 0.790`)**: **`3,434` transactions (`2.91%` of volume)** routed to the human triage queue, strictly satisfying the $\le 3.0\%$ operational capacity constraint (3,543 cases) and capturing **`1,230` true fraud cases** (precision: `35.82%`, `1.79` FP/TP).
  - 🔴 **Auto-Block (`score >= 0.790`)**: **`918` transactions (`0.78%` of volume)** automatically blocked with verified **`90.52%` precision** (stopping **`831` high-confidence frauds**).
  - 🛡️ **Total Fraud Capture**: **`2,061` / `4,064` fraud cases (`50.71%` gross recall)**.
* **Opaque Feature Transparency Protocol**:
  - Plain-language audit reasons are strictly derived from our **20 verified, engineered domain features**.
  - Attaches an explicit **Opaque Signal Contribution metric** disclosing when decisions are heavily weighted by Vesta's undisclosed proprietary features (`V1`–`V339`) without inventing unverified semantic narratives.

---

## 📝 "What Didn't Work" Registry & Lessons Learned

| Attempted Approach | Outcome | Root Cause Analysis & Empirical Pivot |
|---|---|---|
| **Random K-Fold / Shuffling** | *Rejected* | Random splits leak future card velocities and identity fingerprints into past predictions. Replaced with strict 80th-percentile time split. |
| **Static Per-Card Historical Mean** | *Refactored* | Static train-wide averages allowed $t=100$ transactions to reference spend behavior from $t=5000$. Refactored to expanding cumulative mean up to $t-1$. |
| **Independent Per-Split Rolling Windows** | *Refactored* | Resetting velocity windows at the boundary caused early test transactions to register as "first-ever" events. Refactored to streaming state continuity. |
| **Hardcoded `scale_pos_weight = 28.87`** | *Parameterized* | Multiplying positive gradients by ~29 forced trees into noisy leaf splits, degrading PR-AUC from `0.5149` to `0.4629` and Brier score from `0.0224` to `0.1118`. Parameterized as a sweep. |
| **Unconstrained $\tau = 0.010$ Default Policy** | *Refactored* | Flat cost formula mathematical peak required reviewing 51.75% of volume (61,121 cases). Replaced with Capacity-Constrained Primary Policy ($\tau = 0.19$, $\le 3\%$ review budget). |
| **Pure Interpretable-Only Features (20 Signals)** | *Evaluated* | Restricting to 20 engineered features dropped PR-AUC from `0.5149` to `0.2250`. Retained full GBDT with honest Opaque Signal Transparency in Layer 5. |

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

### Running the End-to-End 5-Layer Pipeline

1. **Place IEEE-CIS CSVs in `data/raw/`**:
   - `data/raw/train_transaction.csv`
   - `data/raw/train_identity.csv`

2. **Execute Full Pipeline (Layers 1 to 5)**:
   ```bash
   python3 scripts/run_layer1.py
   python3 scripts/run_layer2.py
   python3 scripts/run_layer3.py
   python3 scripts/run_layer4.py
   python3 scripts/run_layer5.py
   ```

3. **Run Automated Test Suite**:
   ```bash
   python3 -m pytest
   ```

---

## 👥 Authors & Acknowledgments
- **Team**: AI Risk Manager Team
- **Event**: Razorpay AI Buildathon (Track 02)
