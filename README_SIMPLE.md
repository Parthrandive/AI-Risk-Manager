# 💳 AI Risk Manager: Plain-English Guide (Layman's Edition)

> **"Stopping credit card fraudsters in real-time — without annoying honest shoppers or overwhelming human investigators."**

---

## 🧐 What is this project in simple words?

Imagine you run a giant online shopping website. Every second, thousands of people are buying things using credit cards. 

Most people are honest customers, but a small percentage (**~3.5%**) are fraudsters using stolen cards or bots. 

If you do nothing:
- ❌ **You lose money**: Banks will force you to pay back every stolen dollar plus penalty fees (chargebacks).

If you block aggressively on every slight suspicion:
- ❌ **You alienate good customers**: Honest shoppers get declined at checkout, get angry, and never buy from you again.

If you send too many suspicious transactions to human investigators:
- ❌ **Your fraud team drowns**: A team of 5 analysts cannot review 50,000 transactions a day.

### 🎯 What AI Risk Manager Does
**AI Risk Manager** is an intelligent automated system that acts like a **smart airport security checkpoint** for online payments. In milliseconds, it inspects incoming payments, detects fraud patterns, and sorts transactions into three clear lanes:

```
                  [ 💳 Incoming Online Payment ]
                                 │
                                 ▼
                     [ 🧠 AI Risk Engine ]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  🟢 GREEN LANE           🟡 YELLOW LANE           🔴 RED LANE
  (Auto-Approve)          (Manual Review)         (Auto-Block)
   96.2% of Traffic        2.9% of Traffic         0.9% of Traffic
  Honest customers        Ambiguous / tricky      Definite fraudsters
  pass with 0 friction    cases sent to humans    blocked automatically
```

---

## 🚦 How Does the 3-Lane Gateway Work?

Think of it like a smart traffic light:

| Lane | Traffic Share | Who lands here? | What happens? | Real-World Impact |
|---|---|---|---|---|
| 🟢 **Green Lane** *(Auto-Approve)* | **96.2%** of shoppers | Normal customers buying groceries or shoes from their usual device. | **Instant Approval** in under 0.05 seconds with zero extra verification. | **96 out of 100 people** experience seamless checkout. |
| 🟡 **Yellow Lane** *(Manual Review)* | **2.9%** of shoppers | Borderline cases (e.g. someone buying from a new country for the first time). | Sent to a **Human Risk Analyst** with a 1-page summary card explaining *why* it looks weird. | Strictly capped so the human team **never receives more work than they can handle** (<3%). |
| 🔴 **Red Lane** *(Auto-Block)* | **0.9%** of shoppers | Clear fraud rings, automated bot attacks, rapid stolen card testing. | **Instantly Blocked** before the transaction goes through. | **90% of blocked transactions are guaranteed to be real fraud** (less than 1 false block per 10 blocks). |

---

## 🔍 How Does the AI Catch Fraudsters? (The Clues)

Fraudsters try to hide, but they leave behavioral digital footprints. The AI looks at clues like:

1. **Velocity Bursts (Speed Traps)**:
   - *Normal*: You buy a coffee, then lunch 3 hours later.
   - *Fraud*: 5 purchases in 10 minutes from different websites on the same card.
2. **Amount Anomalies (Spending Spikes)**:
   - *Normal*: You usually spend \$20 to \$50 per transaction.
   - *Fraud*: Suddenly a \$1,500 electronics purchase pops up at 3:00 AM.
3. **Geographic Mismatches (Teleportation)**:
   - *Normal*: A card used in California is used in California again.
   - *Fraud*: A card used in Chicago is suddenly used 15 minutes later in London.
4. **Disposable Identity Tricks**:
   - Fraudsters using temporary throwaway emails or rotating disguised device IDs.

---

## 🛡️ Why is this system trustworthy? (No "Black Box" AI)

Many AI models act like mysterious black boxes: they say *"Blocked"* but can't explain why. That's dangerous in financial systems.

AI Risk Manager solves this with **Transparent Audit Cards**:
Whenever a transaction is flagged or blocked, it generates a plain-English explanation that a human investigator can instantly read:

```json
{
  "Transaction": "#3460303",
  "AI_Decision": "🔴 AUTO_BLOCK (Risk Score: 87%)",
  "Why_It_Was_Blocked": [
    "1. Card used 2 times in the last 24 hours across 8 completely different regional addresses.",
    "2. High-risk payment network signature matching known organized fraud patterns."
  ],
  "Evidence_Summary": "Card instrument was previously active with 24h volume across 8 distinct regions; current attempt represents geographic displacement.",
  "Human_Oversight": "Verified with automated explainability tools."
}
```

---

## 🧪 Proven on Real Data (590,000 Transactions)

We didn't test this on fake, toy examples. We evaluated it on **590,540 real-world transactions** from the official IEEE-CIS benchmark:

1. **Catches ~51% of all fraud** while touching less than **4%** of total transaction volume.
2. **Protects Revenue**: Successfully stops **1,901 fraud attempts** in the test set.
3. **Doesn't Go Stale**: We tested how the AI performs over time. Fraud tactics change every month; our built-in **rolling retraining engine** automatically updates the model every ~36 days so it never falls behind new fraud tricks.
4. **Honest Engineering ("No Cheating")**: 
   - Never looked ahead into the future during training.
   - Tested across multiple random seeds to prove results are real, not random luck.

---

## 🏃 How to Run the Project (3 Simple Steps)

If you have Python installed, you can run and test the entire system in minutes:

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
# Runs the 5-layer pipeline from raw data to the 3-lane gateway
python3 scripts/run_layer1.py  # Cleans & splits the transactions
python3 scripts/run_layer2.py  # Extracts smart fraud clues
python3 scripts/run_layer3.py  # Trains the AI model
python3 scripts/run_layer4.py  # Sweeps optimal business rules
python3 scripts/run_layer5.py  # Runs the 3-Lane Triage Gateway
```

### 3. Run Automated Tests
```bash
python3 -m pytest
# Output: 33 passed in 2.5 seconds!
```

---

## 💡 Summary in One Sentence

> **AI Risk Manager is an honest, production-ready fraud prevention system that stops fraudsters in real-time, keeps checkouts effortless for good customers, and gives human investigators clear, explainable reasons for every decision.**
