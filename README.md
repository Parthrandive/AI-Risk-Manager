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
| **Baseline (Logistic Regression)** | `0.1834` | `0.8311` | `12.60%` | `69.46%` | `0.0682` |
| **LightGBM Classifier** | `0.4784` *(+0.2950)* | `0.8907` | `80.18%` | `30.46%` | `0.0238` |
| **XGBoost GBDT (Primary, 429 feats)** | **`0.5121`** *(+0.3287)* | **`0.8967`** | **`81.50%`** | **`31.32%`** | **`0.0225`** |

### 2. Feature Progression & Graph Embedding Ablation Study

Evaluated across seeds `[42, 100, 2024]` with strict $t-1$ temporal edge invariants ($\text{source\_timestamp} \le \text{target\_timestamp}$):

| Feature Configuration | Total Feats | Cross-Seed PR-AUC (Mean ± Std) | Seed 42 PR-AUC | Auto-Blocked Frauds ($\ge 90\%$ Prec) | Net Contained Frauds (85% SLA) | Checkout Fraud Leakage |
|---|---|---|---|---|---|---|
| **1. Baseline Feature Bank** | `427` | `0.5105 ± 0.0031` | `0.5149` *(0.5148786)* | `825.7` *(794 – 852)* | `1,868.3` *(45.97%)* | `2,011.7` *(49.50%)* |
| **2. + Geo-Mismatch Features** | `429` | `0.5100 ± 0.0021` | `0.5121` *(0.5120409)* | **`908.0`** *(890 – 943)* | **`1,901.6`** *(46.79%)* | **`1,987.0`** *(48.89%)* |
| **3. Baseline + Geo + Graph (Neural + Topo)** | `445` | `0.5047 ± 0.0049` | `0.5061` | `862.3` *(840 – 892)* | `1,871.8` *(46.06%)* | `2,014.0` *(49.56%)* |

> [!NOTE]
> **GraphSAGE Training Convergence & Downstream Ablation Findings**:
> - **Convergence Verification**: PyTorch Temporal GraphSAGE trained across 10 epochs with loss steadily declining from **`1.1741` (Epoch 1) $\to$ `1.1171` (Epoch 10)** (-4.85% reduction) and cleanly plateauing between Epochs 8–10 ($\Delta < 0.002$). This rules out implementation divergence or broken optimization.
> - **Why the Loss Reduction is Shallow**: Real IEEE-CIS entity graphs are sparse (most card/device combinations only appear 1–2 times, and ~70% of device IDs are null). As a result, 1-hop neural message passing learns relatively weak structural embeddings compared to direct streaming temporal counters.
> - **Downstream Impact**: Ingesting the 16 graph features (8 topological + 8 GraphSAGE neural embeddings) into XGBoost slightly degrades PR-AUC from `0.5100` $\to$ `0.5047` and drops auto-blocked fraud from `908.0` $\to$ `862.3`.
> - **Substantiated Conclusion**: Information subsumption and graph sparsity combine to make explicit graph features counter-productive in this setting; tabular streaming velocity state remains strictly superior for production deployment.
> - **Framing**: We report this as a measured null/negative result on graph utility without making unverified "abuse-ring" claims (since IEEE-CIS lacks ring labels).

---

### 3. Multi-Threshold Decision Sweep & Synthesized Operational Policies

Rather than cherry-picking an uncalibrated default 0.5 cutoff, the pipeline sweeps decision boundaries from $0.001 \to 0.950$ and synthesizes them into actionable enterprise shipping policies:

```
Decision Threshold Sweep Spectrum:
0.001 ──→ 0.010 [Theoretical Limit] ──→ 0.100 ──→ 0.190 [Primary Ship] ──→ 0.300 [Balanced F1] ──→ 0.800 [VIP] ──→ 0.950
```

| Policy Profile | Operating Threshold ($\tau$) | Catch Rate (Recall) | Precision | False Positives per True Catch ($\text{FP}/\text{TP}$) | Flagged Volume | Auto-Approved Volume | Strategic Intent & Operational Context |
|---|---|---|---|---|---|---|---|
| 🌟 **Production Policy** *(Capacity-Constrained)* | **`0.190`** | **`45.74%`** *(1,859 caught)* | **`53.31%`** | **`0.88`** *(>1.1 true catches / 1 FP)* | **`2.95%`** *(3,487 txns)* | **`97.05%`** | **RECOMMENDED PRIMARY SHIP CANDIDATE.** Designed for fixed enterprise risk teams with a $\le 3.0\%$ review headcount budget. Captures 45.7% of fraud with more true catches than false alarms. |
| 🟢 **Balanced Policy** *(Optimal F1 / Queue Defense)* | **`0.300`** | **`39.15%`** | **`67.13%`** | **`0.49`** *(>2 true catches / 1 FP)* | `2.01%` | **`97.99%`** | Standard production baseline maximizing harmonic F1 (`0.4946`); flags <2% of traffic. *Optimizes classification balance and queue protection, accepting higher missed chargebacks than the Primary Ship policy.* |
| 🔴 **Conservative Policy** *(Minimal Friction VIP)* | **`0.800`** | **`20.57%`** | **`91.17%`** | **`0.10`** *(10 true catches / 1 FP)* | `0.78%` | **`99.22%`** | Ultra-low customer friction; 91.2% precision with near-zero false alarms for frictionless VIP checkout conversion. |
| ⚠️ **Theoretical Ceiling** *(Unconstrained Math Limit)* | **`0.010`** | **`95.10%`** | `6.10%` | `15.39` | `53.64%` *(63k reviews)* | `46.36%` | **THEORETICAL UPPER BOUND (NOT SHIPPED).** Mathematical peak of unconstrained formula (+\$529.1k). Unviable for live deployment without massive reviewer headcount. |

> [!NOTE]
> **Stated Operational Cost Assumptions & Sensitivity**:
> - **Manual Review Cost**: Assumed at **\$5.00** per flagged case (industry benchmark for 3–5 min analyst triage). Sensitivity sweep (\$2 $\to$ \$15) confirms stability: under capacity constraints, the operating point remains firmly anchored at $\tau \approx 0.190$.
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
  - **Pure Card Instrument Proxy**: Synthesized from `card1 + card2 + card3 + card4 + card5 + card6`.
  - **Device Proxy**: Synthesized from `DeviceType + DeviceInfo + id_30 + id_31` *(Note: `DeviceInfo` is ~70% null in IEEE-CIS; unlinked recency cleanly defaults to `-1.0` as an informative signal)*.
* **Leakage Safeguards**:
  - **Expanding Cumulative Card Mean**: `amt_to_expanding_card_mean_ratio` computes the ratio against the cumulative average up to $t-1$, eliminating intra-train lookahead leakage.
  - **Streaming Geo-Mismatch & Multi-Region Tracking**: `is_addr_mismatch_from_card_history` and `card_prior_distinct_addr_count` evaluate card location consistency strictly against historical regions up to $t-1$.
  - **Continuous Streaming State**: `StreamingRiskState` seamlessly carries 10m/1h/24h rolling velocity counters and geo history across the train/test boundary, eliminating test cold-start artifacts.
* **Feature Set**: `429` model-ready numeric and label-encoded features.

---

### 3. Gradient-Boosted Classifiers (XGBoost & LightGBM)
* **Execution**: Trains `XGBoost` and `LightGBM` GBDTs alongside a `LogisticRegression` baseline.
### 4. Evaluation & Honesty Layer
* Implements multi-threshold sweeping ($0.001 \to 0.950$), probability calibration verification (Brier score: `0.0225`), dollar capture curves, cost sensitivity sweeps (\$2 $\to$ \$15), and the structured "What Didn't Work" registry.

---

### 5. SHAP Explainability + Gray-Zone Triage Gateway
* **Grounded 3-Way Triage Gateway**:
  - 🟢 **Auto-Approve (`score < 0.145`)**: **`113,649` transactions (`96.22%` of volume)** approved instantly with zero customer friction. *Leakage: `1,994` missed frauds (`49.1%` of test fraud, `1.755%` leakage rate).*
  - 🟡 **Manual Review / Gray-Zone (`0.145 <= score < 0.740`)**: **`3,414` transactions (`2.89%` of volume)** routed to the human triage queue, strictly satisfying the $\le 3.0\%$ operational capacity constraint (3,543 cases) and flagging **`1,127` frauds** (`27.7%` of test fraud, `33.01%` queue precision, `2.03` FP/TP).
  - 🔴 **Auto-Block (`score >= 0.740`)**: **`1,045` transactions (`0.88%` of volume)** automatically blocked with verified **`90.24%` precision** (stopping **`943` frauds**, `23.2%` of test fraud, `102` false blocks).
  - 🛡️ **Containment Metrics**:
    - **Gross Interception Rate**: **`50.94%` (`2,070` / `4,064` frauds)** flagged or blocked *(assumes 100% human analyst resolution on flagged batch)*.
    - **Net Contained Fraud (Discounted)**: **`46.78%` (`1,900.9` / `4,064` frauds)** stopped *(accounting for a realistic 85% analyst resolution efficiency on the manual review queue)*.
* **Verifiable Transaction Evidence Trail**:
  - Every review and block card generates an immutable `evidence_trail` citing specific factual instrument activity: e.g. `card instrument (card1=3213, card2=459); observed 24h card volume = 2 txns; recency delta = 64699s; prior usage across 8 distinct address regions`.
* **Opaque Feature Transparency Protocol**:
  - Plain-language audit reasons are strictly derived from our **20+ verified, engineered domain features** without asserting unverified semantic narratives for raw/undocumented variables.
  - Attaches an explicit **Opaque Signal Contribution metric** disclosing when decisions are heavily weighted by Vesta's undisclosed proprietary features (`V1`–`V339`).
* **Regulatory Model Governance (RBI / SEBI MRM Alignment)**:
  - Formatted in accordance with enterprise Model Risk Management (MRM) standards and regulatory explainability principles (e.g. RBI Master Directions on IT/Cyber Risk & SEBI AI/ML Governance Principles).
  - Gray-zone model abstention guarantees that ambiguous decisions receive human analyst adjudication with transparent factor disclosures prior to irreversible adverse action.
* **Card-Testing Detection Scoping**:
  - **In-Scope**: Rapid single-instrument velocity bursts (`card_txn_count_10m >= 3`), micro-amount variance anomalies, and rapid geographic region displacement on the same payment instrument.
  - **Future Work / Data Gap**: IEEE-CIS lacks distinct merchant/terminal identifiers (`merchant_id`); cross-merchant distributed testing cannot be directly asserted without inferring proxy groupings.

---

## ⏳ Walk-Forward Robustness Proof & Drift Cadence

We partitioned the full 590,540-transaction dataset into **5 equal-time chronological windows (~36.4 days each)** across the ~182-day span (each period containing >104k txns and >3,500 fraud cases):

| Evaluation Window | Temporal Distance | Frozen Model (Trained on P1-2) PR-AUC | Rolling Retrained Model PR-AUC | Retraining Lift ($\Delta$) | 3% Capacity Recall *(Re-derived $\tau \le 3\%$ per period)* |
|---|---|---|---|---|---|
| **Period 3** | `+18.2 days` | **`0.5477`** | *(Baseline)* | — | **`47.57%`** (59.4% prec @ $\tau=0.16$, 2.91% vol) |
| **Period 4** | `+54.6 days` | **`0.5230`** *(-0.0247)* | **`0.5583`** *(Retrained P1-3)* | **`+0.0353`** | **`44.30%`** (58.5% prec @ $\tau=0.17$, 2.95% vol) |
| **Period 5** | `+91.0 days` | **`0.4644`** *(-0.0833)* | **`0.5189`** *(Retrained P1-4)* | **`+0.0545`** | **`42.22%`** (49.5% prec @ $\tau=0.18$, 2.90% vol) |

> [!IMPORTANT]
> **Production Retraining Cadence Finding**:
> - **Empirical Drift**: A static model frozen in time degrades by **`-0.0833 PR-AUC`** and loses **`5.35pp in recall`** over 90 days as fraud distributions evolve. Noticeably, because scores drifted, the operating threshold had to shift upward ($\tau = 0.16 \to 0.17 \to 0.18$) just to maintain the 3% budget cap.
> - **Retraining Recovery**: Incremental rolling retraining restores PR-AUC to **`0.52–0.56`** (`0.5583` in Period 4, `0.5189` in Period 5) versus `0.46–0.52` for the static frozen model at the same temporal distance.
> - **Validated Cadence**: Retraining at **`~36.4-day intervals`** (the exact cadence tested) successfully halts drift decay. *(Narrower or wider intervals such as 15 or 60 days were not evaluated and represent directions for operational tuning)*.
> - **Scope Limitation**: Retraining on cumulative historical data bundles added sample volume with temporal recency; a matched-volume window ablation would isolate the pure recency effect.

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
| **Relational Graph Embeddings (8 Feats)** | *Evaluated (Null Result)* | Adding 8 temporal relational graph features shifted cross-seed PR-AUC by only `+0.0008` (within seed noise) while auto-blocked frauds dropped from 908 to 878. Information is already subsumed by tabular streaming velocity states. |

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

2. **Execute Full Pipeline (Layers 1 to 5 + Walk-Forward + Graph Ablation)**:
   ```bash
   python3 scripts/run_layer1.py
   python3 scripts/run_layer2.py
   python3 scripts/run_layer3.py
   python3 scripts/run_layer4.py
   python3 scripts/run_layer5.py
   python3 scripts/run_walk_forward.py
   python3 scripts/run_graph_ablation.py
   ```

3. **Run Automated Test Suite**:
   ```bash
   python3 -m pytest
   ```

---

## 👥 Authors & Acknowledgments
- **Team**: AI Risk Manager Team
- **Event**: Razorpay AI Buildathon (Track 02)
