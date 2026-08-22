# 💳 AI Risk Manager: Plain-English Guide (Layman's Edition)

> **"Stopping credit card fraudsters in real time — without annoying honest shoppers or overwhelming human investigators."**  
> *A complete, non-technical translation of the entire AI Risk Manager architecture, benchmark results, deep-dive experiments, and lessons learned.*

---

## 📖 Table of Contents
1. [The Real-World Problem: The Payments Trilemma](#-the-real-world-problem-the-payments-trilemma)
2. [How It Works: The 5-Layer System Explained Simply](#-how-it-works-the-5-layer-system-explained-simply)
3. [The Scorecard: Proven on 590,000 Real Transactions](#-the-scorecard-proven-on-590000-real-transactions)
4. [The 3-Lane Traffic Light Gateway (Live Results)](#-the-3-lane-traffic-light-gateway-live-results)
5. [🔬 The 3 Big Deep-Dive Discoveries](#-the-3-big-deep-dive-discoveries)
   - [Deep Dive 1: The Geographic "Teleportation" Clue](#deep-dive-1-the-geographic-teleportation-clue-step-1)
   - [Deep Dive 2: The Time Machine Test (Why AI Gets Outdated & How We Fix It)](#deep-dive-2-the-time-machine-test-why-ai-gets-outdated--how-we-fix-it-step-2)
   - [Deep Dive 3: The Complex Graph AI Experiment (Why Fancy Graphs Failed)](#deep-dive-3-the-complex-graph-ai-experiment-why-fancy-graphs-failed-step-3)
6. [📋 A Look Inside a Live "Explainable Audit Card"](#-a-look-inside-a-live-explainable-audit-card)
7. [📝 The "What Didn't Work" Registry (7 Honest Lessons Learned)](#-the-what-didnt-work-registry-7-honest-lessons-learned)
8. [🏃 Quickstart: Run & Test in 3 Simple Steps](#-quickstart-run--test-in-3-simple-steps)

---

## 🎯 The Real-World Problem: The Payments Trilemma

When you buy something online with a credit card, the merchant has less than **100 milliseconds** to decide whether to accept your payment. 

Around **3.5%** of all attempted online purchases are illegal fraud (stolen cards, bot attacks, identity theft). 

Every payment company faces a painful **Three-Way Tug of War**:

```
                       [ 💰 Financial Security ]
                       (Catch every stolen dollar)
                                  ▲
                                 / \
                                /   \
                               /     \
                              /       \
  [ 🛍️ Frictionless Checkout ] ◄───────► [ 👥 Human Capacity ]
  (Never annoy honest shoppers)         (Don't drown fraud analysts)
```

1. **If you do nothing**: Fraudsters steal merchandise, and banks hit you with massive penalty chargeback fees.
2. **If you block aggressively on every slight suspicion**: Honest shoppers get falsely declined at checkout, get frustrated, and leave for a competitor.
3. **If you send every suspicious transaction to human investigators**: Your 5-person fraud team drowns under 50,000 cases a day.

### 💡 The Solution: AI Risk Manager
**AI Risk Manager** is an intelligent automated system that acts like a **smart airport security checkpoint** for online payments. It balances this trilemma mathematically:
* It lets **96.2% of honest customers glide through instantly with zero friction**.
* It caps human review strictly to **what analysts can realistically handle (<3.0% of traffic)**.
* It automatically blocks high-confidence fraud with **>90% verified precision**.

---

## 🏗️ How It Works: The 5-Layer System Explained Simply

```
Incoming Payment ──▶ [1. Clean & Sort] ──▶ [2. Smart Clues] ──▶ [3. AI Engine] ──▶ [4. Business Dial] ──▶ [5. 3-Lane Traffic Light]
```

### Layer 1: Ingestion & The "No Peeking" Time Machine Rule
* **What it does**: Ingests 590,540 real transactions and splits them chronologically (the first 80% is the past used for training; the last 20% is the future held out for testing).
* **Why it matters**: In real life, you cannot predict today's fraud using tomorrow's data. Standard AI projects often randomly shuffle data, which is essentially **peeking into the future to cheat on the exam**. We enforce a strict timeline cut so our AI is tested under authentic real-world conditions.

### Layer 2: Extracting Smart Fraud Clues (Feature Engineering)
* The AI computes **429 behavioral clues** for every transaction forward in time without future lookahead:
  1. **Speed Traps (Velocity)**: How many times has this card been swiped in the last 10 minutes, 1 hour, or 24 hours?
  2. **Habit Baselines (Expanding Means)**: If a customer normally spends \$30 on lunch and suddenly attempts a \$2,500 jewelry purchase at 3:00 AM, that's an anomaly.
  3. **Geographic Teleportation**: Has this card suddenly popped up in a new regional location that contradicts its prior history?
  4. **Device Continuity**: Is the transaction coming from a known phone/laptop or an anonymous disposable emulator?

### Layer 3: The AI Brain (Comparing 3 Different Models)
* We pit 3 different AI algorithms against each other:
  1. *Simple Traditional Baseline (Logistic Regression)*: Fast but misses subtle multi-clue combinations.
  2. *LightGBM Classifier*: A fast tree-based model.
  3. *XGBoost Gradient Boosted Trees (Primary Champion)*: Combines hundreds of specialized decision trees to spot hidden fraud patterns.

### Layer 4: The Business Tuning Dial (Threshold Calibration)
* Most naive AI projects use a rigid `0.5` (50%) cutoff. In financial risk, that's reckless!
* Layer 4 sweeps thousands of cutoffs from `0.001` to `0.950` to find the exact operating point that maximizes fraud capture while strictly respecting the human team's review budget.

### Layer 5: The 3-Lane Traffic Light & Explainable Audit Cards
* Instead of a binary "Yes/No", the system routes traffic into **3 operational tiers** and generates a **plain-English explanation card** for every flagged case so human analysts understand *why* the AI made its decision.

---

## 🏆 The Scorecard: Proven on 590,000 Real Transactions

We evaluated our primary model on **118,108 held-out future transactions** (containing 4,064 real fraud attacks):

### 1. Model Performance vs. Baseline & The Kaggle Leaderboard

| AI Architecture | PR-AUC ("Needle in a Haystack" Score) | ROC-AUC | Precision (Accuracy when flagging) | Calibration Brier Loss (Honesty Score) |
|---|---|---|---|---|
| **Simple Baseline (Logistic Regression)** | `0.1834` | `0.8311` | `12.60%` | `0.0682` |
| **LightGBM Classifier** | `0.4784` *(+0.2950)* | `0.8907` | `80.18%` | `0.0238` |
| **XGBoost Primary Champion (429 Clues)** | **`0.5121`** `[0.4971, 0.5276]` | **`0.8967`** `[0.8915, 0.9021]` | **`81.50%`** | **`0.0225`** *(Best)* |
| *Kaggle Competition Top Leaderboard (Offline SOTA)* | *~0.75+ (estimated)* | *`0.9600 – 0.9800`* | *N/A (Multi-model ensemble)* | *N/A* |

> **🏎️ Formula 1 Prototype vs. Fast Production Sports Car**:
> - **Kaggle Top Leaderboard (0.96–0.98 ROC-AUC)**: Built like a Formula 1 racing prototype — they glued together 50+ massive AI models, took hours to calculate global averages across future and past data, and used offline tricks unviable for live websites.
> - **AI Risk Manager (0.8967 ROC-AUC / 0.5121 PR-AUC)**: Built like a high-performance production car. It makes decisions in **under 15 milliseconds**, strictly obeys the timeline without cheating on future data, explains its reasoning to humans, and fits within real operational budgets.

---

### 2. Methodological Rigor: Independent Practice Exams & 1,000 Dice Rolls

To prove our numbers aren't just lucky flukes:
1. **Independent 3-Way Split (Train 70% $\to$ Practice Exam 10% $\to$ Final Test 20%)**:
   - We picked our business rules on the **Practice Exam (Validation set)**, then locked them in and applied them to the **Final Test set touched exactly once**. It delivered **`51.99%` Gross Interception** and **`47.76%` Net Containment**.
2. **1,000-Iteration Bootstrap Confidence Intervals**:
   - We resampled the test data 1,000 times to simulate 1,000 different business scenarios. The AI maintained **`90.30%` block precision (95% range: 88.5% to 92.1%)** and **`0.5125` PR-AUC (95% range: 0.497 to 0.528)**.

---

### 3. Financial Dollar Impact: The Bottom Line ($+\$352,859.38 Net Value)

On the **\$16.24 Million** held-out test transactions, here is the exact dollar impact:

| Business Metric | What It Represents | Dollar Value |
|---|---|---|
| **Total Test Shopping Volume** | All transactions processed in the test period | **`$16,243,432.00`** |
| **Total Fraud Attempted** | The stolen dollars criminals attempted to charge | **`$609,934.31`** |
| **Fraud Losses Prevented** | Stolen goods + bank penalty fees stopped (1.5x goods loss) | **`+$370,907.44`** |
| **Fraud Analyst Labor Cost** | Paying investigators (\$5.00 for 3–5 min review on 3,414 cases) | **`-$17,070.00`** |
| **False Block Friction Cost** | Lost 10% profit margin on 102 mistakenly blocked shoppers | **`-$978.05`** |
| ⭐ **NET FINANCIAL VALUE DELIVERED** | **Real Dollars Saved - Labor - Customer Friction** | **`+$352,859.38`** *(20.7x return on analyst cost)* |

---

## 🚦 The 3-Lane Traffic Light Gateway (Live Results)

Rather than forcing human analysts to drown in cases, our Layer 5 gateway sorts transactions with surgical precision:

| Traffic Lane | Volume | Share % | Fraud Impact | What Happens to the Customer? |
|---|---|---|---|---|
| 🟢 **Green Lane** *(Auto-Approve)* | `113,649` | **`96.22%`** | `1,994` Missed *(1.755% leakage rate)* | **Instant Checkout with Zero Friction.** 96 out of 100 customers experience effortless approval. |
| 🟡 **Yellow Lane** *(Manual Review Queue)* | `3,414` | **`2.89%`** | **`1,127` Frauds Flagged** *(33.01% queue precision)* | **Model Abstains on Ambiguous Cases.** Routed to human risk analysts with a 1-page plain-English explanation card. |
| 🔴 **Red Lane** *(Auto-Block)* | `1,045` | **`0.88%`** | **`943` Frauds Blocked** *(90.24% block precision)* | **Instant Automated Block.** High-confidence fraudsters are stopped immediately. Only 102 false alarms out of 1,045 blocks. |

### 🛡️ Overall Fraud Containment
* **Gross Fraud Intercepted**: **`50.94%` (2,070 out of 4,064 frauds)** flagged or blocked *(assuming 100% analyst catch on flagged batch)*.
* **Net Fraud Contained**: **`46.78%` (1,901.0 frauds)** stopped *(accounting for a realistic 85% analyst resolution efficiency on the human review queue)*.

---

## 🔬 The 3 Big Deep-Dive Discoveries

### Deep Dive 1: The Geographic "Teleportation" Clue (Step 1)

* **The Idea**: If a card is normally used in Chicago, but suddenly attempts a transaction from London 20 minutes later, that's geographic displacement.
* **How We Built It**: We engineered two leak-free streaming features:
  1. `is_addr_mismatch_from_card_history`: A binary flag for whether the location contradicts the card's prior history.
  2. `card_prior_distinct_addr_count`: A counter of how many unique regions the card has been used in up to time $t-1$.
* **The Reality Check on Card #3460024**:
  - We found a card that had been used across **29 distinct regional codes**.
  - *Was this a computer bug or missing data inflating the count?* We pulled the card's full history: it had **2,531 real transactions** with only 4 missing values. It was a genuine high-volume corporate/fleet debit card!
* **The Proven Impact**:
  - Adding these geo clues boosted automatically blocked fraud by **`+82.3 extra frauds (+13.5%)`** across multiple random seeds while maintaining $>90\%$ precision.

---

### Deep Dive 2: The Time Machine Test (Why AI Gets Outdated & How We Fix It) (Step 2)

Fraudsters don't stay still. Once they realize a tactic is getting blocked, they change their tactics, tools, and spending amounts. This is called **Concept Drift**.

To measure how fast an AI model gets "outdated," we split our 6-month dataset into **5 equal-time windows (~36.4 days each)**:

```
[ Period 1 (Month 1) ] ──▶ [ Period 2 (Month 2) ] ──▶ [ Period 3 (Month 3) ] ──▶ [ Period 4 (Month 4) ] ──▶ [ Period 5 (Month 5) ]
|─────── Train AI on Month 1 & 2 ───────|──▶ Test Month 3 ──▶ Test Month 4 (+55d) ──▶ Test Month 5 (+91d)
```

| Evaluation Time Window | Time Since Training | Frozen Old Model (Trained on P1-2) | Rolling Retrained Fresh Model | What We Learned |
|---|---|---|---|---|
| **Period 3** | `+18 days` | **`0.5477`** | *(Baseline)* | Model is fresh and performs at peak accuracy. |
| **Period 4** | `+55 days` | **`0.5230`** *(-0.0247)* | **`0.5583`** *(+0.0353 lift)* | Frozen model begins decaying; fresh retraining completely restores performance. |
| **Period 5** | `+91 days` | **`0.4644`** *(-0.0833 decay)* | **`0.5189`** *(+0.0545 lift)* | **Severe 90-day decay.** The frozen model lost 5.35% in fraud recall. Retraining brought it right back to life! |

> **Takeaway**: A fraud AI left untouched for 3 months degrades significantly. Retraining the model every **~36 days** acts like an "anti-aging vaccine," keeping the AI sharp against modern fraud tricks.

---

### Deep Dive 3: The Complex Graph AI Experiment (Why Fancy Graphs Failed) (Step 3)

* **The Hypothesis**: "What if we connect every card, email, and device in a giant social network web (using deep Graph Neural Networks / GraphSAGE) to catch organized crime rings?"
* **The Strict Invariant**: We built a temporal graph where connections could only look backward in time (never linking to future transactions).
* **Neural Training Convergence**: We trained a PyTorch GraphSAGE neural network across 10 epochs. The training loss dropped steadily from **`1.1741` $\to$ `1.1171`** and cleanly plateaued, proving the neural network trained properly and did not diverge.
* **The Surprising Discovery (*Information Subsumption*)**:
  - When we fed the 16 graph features into our main model, **PR-AUC stayed flat/degraded (`0.5047` vs `0.5100`)** and auto-blocked fraud dropped from `908` to `862`.
  - *Why?* When we looked inside the decision trees, we saw that the AI was using the graph features, but **simultaneously reducing the weight on our tabular speed counters** (`card_txn_count_24h` gain dropped from 23.4 to 16.2).
  - The graph features were simply **duplicating the same clues our fast sliding-window timers had already captured**, while real transaction networks are too sparse (most credit cards appear only once or twice).
* **The Honest Decision**: Rather than keeping a heavy, slow graph model just for marketing hype, **we honestly rejected the graph features for production** and kept our fast, superior 429-feature tabular engine!

---

## 📋 A Look Inside a Live "Explainable Audit Card"

Whenever a transaction lands in the **Manual Review** or **Auto-Block** lane, the system outputs an auditable risk card with zero black-box mystery:

```json
{
  "transaction_id": 3460303,
  "risk_score": 0.8708,
  "decision": "🔴 AUTO_BLOCK",
  "decision_summary": "Score (0.8708) >= 0.740. Transaction automatically blocked with >90% verified precision.",
  "top_plain_english_reasons": [
    "1. Card property attribute carries elevated historical anomaly weight. [SHAP Force: +0.279 log-odds]",
    "2. Payment network signature matches high-risk commercial card cluster. [SHAP Force: +0.245 log-odds]",
    "3. Missing billing address region contributes positive risk weight. [SHAP Force: +0.046 log-odds]"
  ],
  "evidence_trail": {
    "card_identifier": "card1=3213, card2=459",
    "historical_activity_summary": "Card instrument was active in 24h history (2 prior transactions, recency delta = 64699s) across 8 distinct address regions.",
    "is_geographic_mismatch": false
  },
  "opaque_signal_transparency": {
    "vendor_v_feature_contribution_pct": 48.6,
    "disclosure": "48.6% of this decision was influenced by Vesta's proprietary undisclosed features (V1-V339). Verified domain clues above represent the interpretable portion."
  },
  "governance_and_oversight": {
    "principles": "Designed for transparent, auditable decisioning and operational model risk management.",
    "human_in_the_loop_safeguard": "Gray-zone model abstention guarantees human analyst adjudication with verifiable factor cards prior to irreversible action."
  }
}
```

---

## ⏳ Label Maturation: Why Recent Fraud Data is "Still Cooking"

In payment card fraud, truth takes time. When a credit card is stolen:
1. **Day 0**: Criminal buys a laptop on your store. The system sees a normal transaction.
2. **Day 30**: The innocent cardholder opens their monthly bank statement, sees the charge, and panics.
3. **Day 60–120**: The bank completes an investigation and files an official **chargeback** dispute.

> **The Reality**: The most recent 30 to 60 days of data always *undercounts* real fraud because victim cardholders haven't noticed yet. 
> 
> **How We Handle It in Production**: We enforce a **30–60 Day Maturity Buffer** — when training fresh models, we drop the most recent 30–60 days of unconfirmed data so the AI never gets trained on false negatives!

---

## 📝 The "What Didn't Work" Registry (7 Honest Lessons Learned)

True engineering is defined by what you test, measure, and discard. Here are 7 real anti-patterns we identified and solved:

| What Was Attempted | Why It Failed | The Honest Solution We Built |
|---|---|---|
| **1. Random Data Shuffling** | Randomly splitting data leaks future customer behavior into past predictions (cheating). | Replaced with strict **80/20 chronological time split**. |
| **2. Static Average Card Spending** | Calculating an overall average spend per card allows a transaction at $t=1$ to "know" what the customer spent at $t=100$. | Built an **expanding average** that only knows spending up to $t-1$. |
| **3. Resetting Timers at Test Boundary** | If you reset your 24-hour counters when testing begins, the first test transaction looks like a brand-new card with zero history. | Built a **continuous streaming state engine** that carries memory across the boundary. |
| **4. Forcing 29x Fraud Weight (`scale_pos_weight`)** | Artificially forcing the model to treat fraud as 29x heavier caused it to panic and produce massive false alarms. | Parameterized the loss and used natural class weights with post-hoc probability thresholding. |
| **5. The Naive 0.01 Threshold** | Pure math formulas suggested reviewing 51% of all volume (61,000 cases) — which would bankrupt a real fraud team. | Built the **Grounded Capacity Gateway** ($\le 3.0\%$ review budget). |
| **6. Using Only 20 Simple Features** | Restricting the AI strictly to 20 simple features crippled model accuracy (PR-AUC dropped from `0.51` down to `0.22`). | Retained the full 429-feature model and added **Opaque Signal Transparency Disclosures**. |
| **7. Relational Graph Neural Networks (GraphSAGE)** | Complex graph connections duplicated existing streaming timers without adding new signal. | Documented the **empirical null result** and kept the lightweight tabular engine. |

---

## 🏃 Quickstart: Run & Test in 3 Simple Steps

### Step 1: Install Requirements
```bash
git clone git@github.com:Parthrandive/AI-Risk-Manager.git
cd AI-Risk-Manager
pip install -r requirements.txt
```

### Step 2: Run the 5-Layer Pipeline
```bash
python3 scripts/run_layer1.py  # Cleans & splits transactions
python3 scripts/run_layer2.py  # Builds 429 streaming clues
python3 scripts/run_layer3.py  # Trains the AI models
python3 scripts/run_layer4.py  # Sweeps optimal business thresholds
python3 scripts/run_layer5.py  # Runs the 3-Lane Traffic Light Gateway
```

### Step 3: Run the Automated Verification Suite
```bash
python3 -m pytest
# ✔ 33 tests passed in 2.5 seconds!
```

---

## 💡 Summary in One Sentence

> **AI Risk Manager is an honest, mathematically grounded fraud engine that catches ~51% of fraud, protects 96% of honest shoppers from checkout friction, respects human review capacity, and provides crystal-clear reasons for every single decision.**
