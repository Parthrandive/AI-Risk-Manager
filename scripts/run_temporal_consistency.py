"""
Layer 2 Diagnostic: Feature Temporal Consistency & Stationarity Audit
=====================================================================
Razorpay Buildathon — Track 02: AI Risk Manager

Evaluates whether individual feature predictive power holds over time
(trained on Month 1, evaluated on Month 5) or suffers from non-stationary
decay / signal inversion.
"""

import os
import sys
import json
import logging
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import audit_feature_temporal_consistency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAIN_FEATURES_PATH = "data/processed/train_features.parquet"
OUTPUT_DIR = "data/processed"
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "temporal_consistency_report.json")


def main():
    if not os.path.exists(TRAIN_FEATURES_PATH):
        logger.error(f"Transformed train features not found at {TRAIN_FEATURES_PATH}. Please run Layer 2 first.")
        sys.exit(1)

    logger.info(f"Loading {TRAIN_FEATURES_PATH}...")
    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)

    # Key engineered domain features to highlight
    domain_features = [
        "card_txn_count_10m",
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_amt_sum_24h",
        "amt_to_expanding_card_mean_ratio",
        "time_since_last_txn_card",
        "card_prior_distinct_addr_count",
        "is_addr_mismatch_from_card_history",
        "TransactionAmt",
        "is_same_email_domain",
        "is_high_risk_email",
        "ProductCD_encoded",
        "card6_encoded"
    ]

    logger.info("Running comprehensive audit across all 429 features...")
    report = audit_feature_temporal_consistency(train_df)

    # Save full report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {REPORT_OUTPUT_PATH}")

    # Display clean table for engineered features
    df_metrics = pd.DataFrame(report["feature_metrics"])
    df_domain = df_metrics[df_metrics["feature_name"].isin(domain_features)].copy()

    print("\n" + "=" * 80)
    print("=== ENGINEERED DOMAIN FEATURES TEMPORAL CONSISTENCY (Month 1 -> Month 5) ===")
    print("=" * 80)
    print(df_domain.to_string(index=False))

    # Display top decaying vendor features
    df_decaying = df_metrics[df_metrics["stability_status"].isin(["DECAYING", "INVERTED (TOXIC)"])].sort_values("temporal_shift_delta")
    print("\n" + "=" * 80)
    print("=== TOP NON-STATIONARY / COLLAPSING VENDOR FEATURES (Caught by Audit) ===")
    print("=" * 80)
    print(df_decaying.head(10).to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
