#!/usr/bin/env python3
"""
CLI Script to execute Layer 4 (Evaluation & Honesty Layer) of the AI Risk Manager pipeline.
"""

import argparse
import sys
import os
import pandas as pd

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import run_layer4_pipeline, get_what_didnt_work_registry


def main():
    parser = argparse.ArgumentParser(
        description="Layer 4: Multi-threshold precision/recall sweep, false positive cost curves, and 'What Didn't Work' log."
    )
    parser.add_argument(
        "--test-features",
        type=str,
        default="data/processed/test_features.parquet",
        help="Path to test_features.parquet (from Layer 2)"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        default="data/processed/train_features.parquet",
        help="Path to train_features.parquet (optional for weight analysis)"
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
        help="Directory to save evaluation reports"
    )

    args = parser.parse_args()

    if not os.path.exists(args.test_features) or not os.path.exists(args.model_path):
        print("\n❌ Error: Required Layer 2/3 outputs not found.")
        print(f"Missing: {args.test_features} or {args.model_path}")
        print("Please run Layers 2 and 3 first: python3 scripts/run_layer2.py && python3 scripts/run_layer3.py")
        sys.exit(1)

    try:
        results = run_layer4_pipeline(
            test_features_path=args.test_features,
            train_features_path=args.train_features,
            model_path=args.model_path,
            output_dir=args.output_dir
        )

        sweep_csv = results["artifact_paths"]["threshold_sweep_csv"]
        sweep_df = pd.read_csv(sweep_csv)

        print("\n" + "=" * 88)
        print("🚀 LAYER 4: EVALUATION & HONESTY LAYER SUMMARY")
        print("=" * 88)
        print(f"• Total Test Transactions: {results['total_test_transactions']:,}")
        print(f"• True Fraud Cases:        {results['total_test_fraud']:,} ({results['test_fraud_rate']*100:.3f}%)")
        print(f"• Overall PR-AUC:          {results['overall_pr_auc']:.4f}")
        print(f"• Overall ROC-AUC:         {results['overall_roc_auc']:.4f}")
        print(f"• Brier Score Loss:        {results['brier_score_loss']:.4f} (Probability Calibration)")

        print("\n📊 MULTI-THRESHOLD DECISION SWEEP (Granular Operating Points):")
        print(f"{'Thresh':<8} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'FP / TP':<10} | {'Flagged %':<10} | {'Approved %':<10}")
        print("-" * 88)
        
        sample_thresholds = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
        sample_rows = sweep_df[sweep_df["threshold"].isin(sample_thresholds)]
        if sample_rows.empty:
            sample_rows = sweep_df.iloc[::2]

        for _, row in sample_rows.iterrows():
            fp_str = f"{row['flags_per_true_fraud']:.2f}" if row['flags_per_true_fraud'] >= 0 else "N/A"
            print(f"{row['threshold']:<8.3f} | {row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['f1_score']:<8.4f} | {fp_str:<10} | {row['flagged_percentage']:<10.2f} | {row['auto_approved_percentage']:<10.2f}")
        print("-" * 88)

        print("\n🎯 SYNTHESIZED OPERATIONAL POLICIES:")
        policies = results.get("operational_policies", {})
        policy_order = [
            "capacity_constrained_primary",
            "balanced_f1",
            "conservative_vip",
            "unconstrained_theoretical_peak"
        ]
        for p_key in policy_order:
            if p_key in policies:
                p = policies[p_key]
                print(f"\n• {p.get('name', p_key)} [Threshold τ = {p.get('threshold', 0.0):.3f}]:")
                print(f"  - Status:        {p.get('operational_status', '')}")
                print(f"  - Precision:     {p.get('precision', 0.0)*100:.2f}% | Catch Rate (Recall): {p.get('recall', 0.0)*100:.2f}% | FP/TP: {p.get('flags_per_true_fraud', 0.0):.2f}")
                print(f"  - Traffic Split: Flagged: {p.get('flagged_percentage', 0.0):.2f}% | Auto-Approved: {p.get('auto_approved_percentage', 0.0):.2f}%")
                print(f"  - Strategic Rationale: {p.get('rationale', '')}")

        print("\n📝 'WHAT DIDN'T WORK' REGISTRY & LESSONS LEARNED:")
        for idx, entry in enumerate(get_what_didnt_work_registry(), 1):
            print(f"\n{idx}. [{entry['category']}] {entry['hypothesis_or_approach']}")
            print(f"   • Status:   {entry['outcome']}")
            print(f"   • Analysis: {entry['root_cause_analysis']}")

        print("\n📁 GENERATED ARTIFACTS:")
        print(f"• Threshold Sweep (Parquet): {results['artifact_paths']['threshold_sweep_parquet']}")
        print(f"• Threshold Sweep (CSV):     {results['artifact_paths']['threshold_sweep_csv']}")
        print(f"• Evaluation Summary:        {results['artifact_paths']['evaluation_summary']}")
        print(f"• What Didn't Work Log:      {results['artifact_paths']['what_didnt_work']}")
        print("=" * 88 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
