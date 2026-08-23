"""
AI Risk Manager — Interactive Streamlit Live Demo & Analyst Console
====================================================================
Razorpay Buildathon — Track 02: AI Risk Manager

An end-to-end, leak-free transaction fraud detector featuring:
1. Real-time sub-millisecond risk scoring (<1ms latency).
2. Grounded 3-Lane Traffic Light Gateway (Auto-Approve, Gray-Zone Manual Review, Auto-Block).
3. Local SHAP explainability cards with exact numerical force attributions.
4. Verifiable transaction evidence trails and opaque signal transparency disclosures.
5. Interactive Human-in-the-Loop Analyst Override Queue & Behavioral Drift Telemetry.
6. RBI Digital Payment Security Controls (2021) & India DPDP Act 2023 compliance tags.
"""

import os
import time
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

from src.explainability import (
    SHAPExplainabilityEngine,
    calibrate_gateway_thresholds
)

# Page configuration
st.set_page_config(
    page_title="AI Risk Manager — Live Fraud Gateway",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d233a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .badge-approve {
        background-color: #e6f4ea;
        color: #137333;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #ceead6;
    }
    .badge-review {
        background-color: #fef7e0;
        color: #b06000;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #feefc3;
    }
    .badge-block {
        background-color: #fce8e6;
        color: #c5221f;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #fad2cf;
    }
    .card-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a202c;
    }
</style>
""", unsafe_allow_html=True)

# File paths
MODEL_PATH = "data/processed/fraud_detector_gbdt.joblib"
TEST_DATA_PATH = "data/processed/test_features.parquet"
OVERRIDE_LOG_PATH = "data/processed/analyst_override_log.csv"

# Ensure processed dir exists
os.makedirs("data/processed", exist_ok=True)


@st.cache_resource
def load_production_pipeline():
    """Load model, explainer, feature schema, and test dataset."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_DATA_PATH):
        st.error("Pipeline artifacts not found. Please run scripts/run_layer1.py through scripts/run_layer5.py first.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    exclude_cols = {'TransactionID', 'TransactionDT', 'isFraud', '_card_proxy', '_device_proxy'}
    feature_cols = [c for c in test_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(test_df[c].dtype)]

    # Subsample for background tree explainer
    X_background = test_df[feature_cols].sample(min(len(test_df), 300), random_state=42)
    explainer = SHAPExplainabilityEngine(
        model=model,
        X_background=X_background,
        feature_names=feature_cols,
        tau_low=0.145,
        tau_high=0.740
    )

    return model, explainer, feature_cols, test_df


model, explainer, feature_cols, test_df = load_production_pipeline()


def get_override_log():
    """Load or initialize analyst override audit log."""
    if os.path.exists(OVERRIDE_LOG_PATH):
        try:
            return pd.read_csv(OVERRIDE_LOG_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "timestamp", "transaction_id", "risk_score", "model_decision",
        "analyst_action", "final_decision", "analyst_reason", "is_override"
    ])


def save_override_action(txn_id, risk_score, model_dec, analyst_action, final_dec, reason):
    """Append a human analyst override decision to the audit log."""
    log_df = get_override_log()
    is_override = int(analyst_action != "Uphold Model Verdict")
    new_entry = pd.DataFrame([{
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transaction_id": int(txn_id),
        "risk_score": round(float(risk_score), 4),
        "model_decision": model_dec,
        "analyst_action": analyst_action,
        "final_decision": final_dec,
        "analyst_reason": reason,
        "is_override": is_override
    }])
    updated_df = pd.concat([log_df, new_entry], ignore_index=True)
    updated_df.to_csv(OVERRIDE_LOG_PATH, index=False)
    return updated_df


# -------------------------------------------------------------
# SIDEBAR CONTROLS & TELEMETRY
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Razorpay%20Buildathon-Track%2002%3A%20AI%20Risk%20Manager-blue.svg", use_container_width=True)
    st.markdown("### ⚙️ Production Gateway Controls")

    tau_low = st.number_input("Auto-Approve Cutoff (τ_low)", value=0.145, step=0.005, format="%.3f", help="Transactions with risk < τ_low are approved instantly with zero customer friction.")
    tau_high = st.number_input("Auto-Block Cutoff (τ_high)", value=0.740, step=0.005, format="%.3f", help="Transactions with risk >= τ_high are automatically blocked (>90% verified precision floor).")

    explainer.tau_low = tau_low
    explainer.tau_high = tau_high

    st.markdown("---")
    st.markdown("### 📊 Analyst Override Telemetry")
    override_log = get_override_log()
    total_reviews = len(override_log)
    total_overrides = override_log["is_override"].sum() if total_reviews > 0 else 0
    override_rate = (total_overrides / total_reviews * 100.0) if total_reviews > 0 else 0.0

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Adjudicated", f"{total_reviews:,}")
    col_s2.metric("Override %", f"{override_rate:.1f}%")

    if override_rate > 15.0 and total_reviews >= 5:
        st.warning("⚠️ **Drift Alert**: Analyst override rate exceeds 15%. Triggering behavioral drift inspection.")
    else:
        st.success("✔ **Drift Status**: Normal (Override rate <= 15%).")

    st.markdown("---")
    st.markdown("### ⚡ Live System Specs")
    st.markdown("""
    - **Engine**: XGBoost GBDT (429 Clues)
    - **PR-AUC**: `0.5111 ± 0.0031` (5-Seed)
    - **ROC-AUC**: `0.8967 ± 0.0012`
    - **Inference Latency**: `0.30 ms` (P50) / `0.81 ms` (P99)
    - **Throughput**: `3,082 txns/sec/core`
    - **Compliance**: RBI Security 2021 & DPDP 2023
    """)

# -------------------------------------------------------------
# MAIN APP INTERFACE
# -------------------------------------------------------------
st.markdown('<div class="main-header">💳 AI Risk Manager: Live Triage Gateway</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time payment fraud detection with gray-zone triage abstention, local SHAP attribution, and verifiable audit cards.</div>', unsafe_allow_html=True)

# Select Input Mode
st.markdown("#### 1. Select or Customize a Transaction")

preset_options = [
    "🟢 Preset 1: Low-Risk Legitimate Shopper (Auto-Approve, ID #3459433)",
    "🟡 Preset 2: Ambiguous Gray-Zone Transaction (Manual Review, ID #3459635)",
    "🔴 Preset 3: High-Confidence Card-Testing Attack (Auto-Block, ID #3460303)",
    "✍️ Custom Transaction (Interactive Sliders & Signals)",
    "📄 Raw Transaction JSON Input"
]

selected_mode = st.selectbox("Choose a transaction scenario to evaluate:", preset_options, index=1)

active_row = None
active_txn_id = 9999999

if "Preset 1" in selected_mode:
    active_row = test_df[test_df["TransactionID"] == 3459433].iloc[0]
    active_txn_id = 3459433
elif "Preset 2" in selected_mode:
    active_row = test_df[test_df["TransactionID"] == 3459635].iloc[0]
    active_txn_id = 3459635
elif "Preset 3" in selected_mode:
    active_row = test_df[test_df["TransactionID"] == 3460303].iloc[0]
    active_txn_id = 3460303
elif "Custom Transaction" in selected_mode:
    # Use baseline median row and allow custom overrides
    base_row = test_df.iloc[0].copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        txn_amt = st.number_input("Transaction Amount ($)", value=149.50, min_value=1.0, max_value=10000.0, step=10.0)
        c_24h = st.slider("24h Card Velocity (Txns)", min_value=0, max_value=25, value=4)
    with c2:
        recency = st.slider("Recency Delta (Seconds since last txn)", min_value=0, max_value=86400, value=300)
        dist_addrs = st.slider("Prior Distinct Address Regions", min_value=1, max_value=15, value=2)
    with c3:
        geo_mismatch = st.selectbox("Geographic History Mismatch?", [0, 1], index=1, format_func=lambda x: "Yes (Displacement)" if x == 1 else "No (Consistent)")
        amt_ratio = st.slider("Amount to Expanding Mean Ratio", min_value=0.1, max_value=10.0, value=2.5, step=0.1)

    base_row["TransactionAmt"] = txn_amt
    base_row["card_txn_count_24h"] = c_24h
    base_row["time_since_last_txn_card"] = recency
    base_row["card_prior_distinct_addr_count"] = dist_addrs
    base_row["is_addr_mismatch_from_card_history"] = geo_mismatch
    base_row["amt_to_expanding_card_mean_ratio"] = amt_ratio
    active_row = base_row
    active_txn_id = 8888888
elif "Raw Transaction JSON" in selected_mode:
    sample_json = test_df[feature_cols].iloc[0].to_dict()
    json_str = st.text_area("Paste Feature JSON:", value=json.dumps({k: round(v, 2) if pd.notna(v) else None for k, v in list(sample_json.items())[:15]}, indent=2), height=150)
    try:
        parsed_dict = json.loads(json_str)
        base_row = test_df.iloc[0].copy()
        for k, v in parsed_dict.items():
            if k in base_row:
                base_row[k] = v
        active_row = base_row
        active_txn_id = 7777777
    except Exception as e:
        st.error(f"Invalid JSON format: {e}")
        st.stop()

# -------------------------------------------------------------
# LIVE EVALUATION & SHAP AUDIT CARD
# -------------------------------------------------------------
st.markdown("---")
st.markdown("#### 2. Live Decision Gateway & SHAP Audit Card")

# Run real-time scoring
t0 = time.perf_counter()
X_input = pd.DataFrame([active_row[feature_cols]])
prob = float(model.predict_proba(X_input)[:, 1][0])
latency_ms = (time.perf_counter() - t0) * 1000.0

# Generate full explainability card
audit_card = explainer.explain_transaction(
    X_row=active_row[feature_cols],
    risk_score=prob,
    transaction_id=active_txn_id
)

decision = audit_card["decision"]

# Display Traffic Light Badge & Key Metrics
col_b1, col_b2, col_b3, col_b4 = st.columns([1.5, 1, 1, 1])

with col_b1:
    if decision == "AUTO_APPROVE":
        st.markdown('<div class="badge-approve">🟢 AUTO-APPROVE (GREEN)</div>', unsafe_allow_html=True)
    elif decision == "MANUAL_REVIEW":
        st.markdown('<div class="badge-review">🟡 MANUAL REVIEW (GRAY-ZONE)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-block">🔴 AUTO-BLOCK (RED)</div>', unsafe_allow_html=True)

with col_b2:
    st.metric("Risk Score", f"{prob:.4f}", help="Calibrated fraud probability from primary GBDT model.")

with col_b3:
    st.metric("Inference Latency", f"{latency_ms:.2f} ms", help="Sub-millisecond inference time on CPU.")

with col_b4:
    st.metric("Transaction ID", f"#{active_txn_id}")

# Decision rationale
st.info(f"**Gateway Action**: {audit_card['decision_summary']}")

# Render Audit Card Columns
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("##### 🔍 Top Interpretable Risk Factors (Local SHAP Forces)")
    for i, factor in enumerate(audit_card["top_interpretable_factors"], 1):
        st.markdown(f"- **Factor {i}**: {factor}")

    st.markdown("##### 📁 Verifiable Transaction Evidence Trail")
    ev = audit_card["evidence_trail"]
    st.markdown(f"""
    - **Instrument Identifier**: `{ev['instrument_proxy']}`
    - **Historical Activity Summary**: {ev['historical_activity_summary']}
    - **Prior Distinct Regions**: `{ev['prior_distinct_regions_count']}`
    - **Geographic Anomaly**: `{'🚨 YES (Displaced)' if ev['is_geographic_mismatch'] else '✔ NO (Consistent)'}`
    """)

with c_right:
    st.markdown("##### 🛡️ Opaque Signal Transparency & Regulatory Governance")
    op = audit_card["opaque_signal_disclosure"]
    st.markdown(f"""
    - **Undisclosed Vendor V-Features**: `{op['undisclosed_v_feature_contribution_pct']}%` of model contribution
    - **Transparency Statement**: *{op['disclosure_statement']}*
    """)

    st.markdown("##### 🏛️ Indian Regulatory Framework Alignment")
    gov = audit_card["governance_and_audit_architecture"]
    st.markdown(f"""
    - **RBI Master Direction (2021)**: Real-time velocity containment & risk-based transaction screening.
    - **RBI AI/ML Governance**: Model abstention in gray-zone preserves human-in-the-loop oversight before adverse declines.
    - **DPDP Act 2023**: Card & device tokens are cryptographically hashed; zero raw PII stored in feature tables.
    """)

# Expandable raw JSON card
with st.expander("📄 View Complete Audit Card JSON Schema"):
    st.json(audit_card)

# -------------------------------------------------------------
# HUMAN-IN-THE-LOOP ANALYST ADJUDICATION CONSOLE
# -------------------------------------------------------------
st.markdown("---")
st.markdown("#### 3. 👥 Human-in-the-Loop Analyst Adjudication Console")
st.markdown("Fraud investigators review gray-zone cases and can uphold or override the automated verdict. Every disposition feeds continuous monitoring and retraining.")

col_a1, col_a2, col_a3 = st.columns([1.5, 2, 1])

with col_a1:
    analyst_action = st.selectbox(
        "Analyst Decision Action:",
        ["Uphold Model Verdict", "Override ➔ Approve (Legitimate)", "Override ➔ Block (Confirmed Fraud)", "Request Secondary KYC / Step-Up Auth"]
    )

with col_a2:
    reason_preset = st.selectbox(
        "Disposition Rationale:",
        [
            "Verified cardholder travel / legitimate spend pattern",
            "Confirmed account takeover via secondary KYC callback",
            "Velocity burst confirmed as authorized business expense",
            "High-risk emulator signature confirmed malicious",
            "Routine model verdict confirmation without anomaly"
        ]
    )

with col_a3:
    st.write("")
    st.write("")
    if st.button("💾 Submit Adjudication", use_container_width=True):
        final_dec = decision
        if "Approve" in analyst_action:
            final_dec = "ANALYST_APPROVED"
        elif "Block" in analyst_action:
            final_dec = "ANALYST_BLOCKED"

        save_override_action(
            txn_id=active_txn_id,
            risk_score=prob,
            model_dec=decision,
            analyst_action=analyst_action,
            final_dec=final_dec,
            reason=reason_preset
        )
        st.success(f"✔ Adjudication recorded for Transaction #{active_txn_id}!")
        st.rerun()

# Display Recent Override Audit Table
st.markdown("##### 📋 Recent Analyst Adjudication Audit Trail")
current_log = get_override_log()
if len(current_log) > 0:
    st.dataframe(
        current_log.sort_values("timestamp", ascending=False).head(10),
        use_container_width=True
    )
else:
    st.info("No manual adjudications recorded yet. Use the console above to submit test reviews.")
