#!/usr/bin/env python3
"""
CLI Script to execute Layer 5 (SHAP Explainability & Gray-Zone Triage Gateway) of the AI Risk Manager pipeline.
"""

import argparse
import sys
import os
import json

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability import run_layer5_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Layer 5: Calibrate grounded 3-way triage gateway and generate local SHAP audit trails."
    )
    parser.add_argument(
        "--test-features",
        type=str,
        default="data/processed/test_features.parquet",
        help="Path to test_features.parquet (from Layer 2)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/processed/fraud_detector_gbdt.joblib",
        help="Path to trained GBDT model (from Layer 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save triage artifacts"
    )
    parser.add_argument(
        "--review-budget-pct",
        type=float,
        default=3.0,
        help="Maximum operational review queue capacity percentage (default: 3.0%%)"
    )
    parser.add_argument(
        "--min-autoblock-precision",
        type=float,
        default=90.0,
        help="Minimum verified precision floor required for automated blocking (default: 90.0%%)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.test_features) or not os.path.exists(args.model_path):
        print("\n❌ Error: Required Layer 2/3 outputs not found.")
        print(f"Missing: {args.test_features} or {args.model_path}")
        print("Please run Layers 2 and 3 first: python3 scripts/run_layer2.py && python3 scripts/run_layer3.py")
        sys.exit(1)

    try:
        results = run_layer5_pipeline(
            test_features_path=args.test_features,
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_review_budget_pct=args.review_budget_pct,
            min_autoblock_precision=args.min_autoblock_precision
        )

        gw = results["gateway_thresholds"]
        dist = results["triage_distribution"]
        contain = results["overall_fraud_containment"]

        print("\n" + "=" * 88)
        print("🚀 LAYER 5: SHAP EXPLAINABILITY & GRAY-ZONE TRIAGE GATEWAY")
        print("=" * 88)
        print(f"• Total Evaluated Transactions: {results['total_evaluated_transactions']:,}")
        print(f"• Grounded Gateway Boundaries:   Auto-Approve (< {gw['tau_low_auto_approve']:.3f}) | Review Queue [{gw['tau_low_auto_approve']:.3f}, {gw['tau_high_auto_block']:.3f}) | Auto-Block (>= {gw['tau_high_auto_block']:.3f})")
        print(f"• Grounding Constraints:        Review Budget <= {args.review_budget_pct:.1f}% ({results['calibration_constraints']['max_review_cases_budget']:,} txns) | Auto-Block Precision Floor >= {args.min_autoblock_precision:.1f}%")

        print("\n🚦 THREE-WAY TRIAGE TRAFFIC ROUTING DISTRIBUTION:")
        print("-" * 88)
        print(f"{'Decision Tier':<24} | {'Volume':<10} | {'Share %':<10} | {'Fraud Caught':<14} | {'Precision / Risk':<18}")
        print("-" * 88)
        
        app = dist["auto_approve"]
        rev = dist["manual_review_gray_zone"]
        blk = dist["auto_block"]

        print(f"🟢 {'Auto-Approve':<21} | {app['count']:<10,d} | {app['percentage']:<9.2f}% | {app['fraud_count']:<14,d} | Leakage: {app['leakage_rate']:.3f}%")
        print(f"🟡 {'Manual Review (Gray-Zone)':<21} | {rev['count']:<10,d} | {rev['percentage']:<9.2f}% | {rev['fraud_count']:<14,d} | Precision: {rev['precision']:.2f}% ({rev['flags_per_true_catch']:.2f} FP/TP)")
        print(f"🔴 {'Auto-Block':<21} | {blk['count']:<10,d} | {blk['percentage']:<9.2f}% | {blk['fraud_count']:<14,d} | Precision: {blk['precision']:.2f}% (False Blocks: {blk['false_block_count']})")
        print("-" * 88)
        print(f"🛡️  OVERALL FRAUD CONTAINMENT: {contain['total_caught_fraud']:,} / {contain['total_fraud_cases']:,} Frauds Captured ({contain['gross_recall_pct']:.2f}% Gross Recall)")

        print("\n📋 SAMPLE AUDIT CARDS WITH LOCAL SHAP & OPAQUE SIGNAL TRANSPARENCY:")
        for idx, card in enumerate(results.get("sample_audit_cards", [])[:3], 1):
            tier_symbol = "🟢" if card["decision"] == "AUTO_APPROVE" else ("🟡" if card["decision"] == "MANUAL_REVIEW" else "🔴")
            print(f"\n[{idx}] {tier_symbol} Transaction #{card['transaction_id']} | Risk Score: {card['risk_score']:.4f} -> Decision: {card['decision']}")
            print(f"    • Rationale: {card['decision_summary']}")
            print("    • Top Verified Domain Risk Factors:")
            for factor in card["top_interpretable_factors"]:
                print(f"      - {factor}")
            print(f"    • Opaque Feature Disclosure: {card['opaque_signal_disclosure']['disclosure_statement']}")

        print("\n📁 GENERATED ARTIFACTS:")
        print(f"• Triage Queue (Parquet):   {results['artifact_paths']['triage_queue_parquet']}")
        print(f"• Triage Summary (JSON):    {results['artifact_paths']['triage_summary_json']}")
        print("=" * 88 + "\n")

    except Exception as e:
        print(f"\n❌ Layer 5 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
