#!/usr/bin/env python3
"""
CLI Script to execute Layer 3 (Model Training & Baseline Comparison) of the AI Risk Manager pipeline.
"""

import argparse
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import run_layer3_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Layer 3: Train baseline model and XGBoost GBDT classifier on held-out test split."
    )
    parser.add_argument(
        "--train-features",
        type=str,
        default="data/processed/train_features.parquet",
        help="Path to train_features.parquet (from Layer 2)"
    )
    parser.add_argument(
        "--test-features",
        type=str,
        default="data/processed/test_features.parquet",
        help="Path to test_features.parquet (from Layer 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save model artifacts and metrics.json"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=150,
        help="Number of trees for XGBoost"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max tree depth for XGBoost"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for XGBoost"
    )

    args = parser.parse_args()

    if not os.path.exists(args.train_features) or not os.path.exists(args.test_features):
        print("\n❌ Error: Layer 2 feature files not found.")
        print(f"Missing: {args.train_features} or {args.test_features}")
        print("Please run Layer 2 first: python3 scripts/run_layer2.py")
        sys.exit(1)

    try:
        results = run_layer3_pipeline(
            train_features_path=args.train_features,
            test_features_path=args.test_features,
            output_dir=args.output_dir,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate
        )

        base_m = results["baseline_metrics"]
        gbdt_m = results["gbdt_metrics"]
        comp = results["comparison"]

        print("\n" + "=" * 68)
        print("🚀 LAYER 3: MODEL TRAINING & BASELINE COMPARISON SUMMARY")
        print("=" * 68)
        print(f"• Evaluated Features:  {results['feature_count']}")
        print(f"• Scale Pos Weight:    {gbdt_m.get('scale_pos_weight_used', 1.0):.2f}")
        
        print("\n📊 TEST SET PERFORMANCE COMPARISON:")
        print(f"{'Metric':<20} | {'Baseline (LogReg)':<18} | {'XGBoost GBDT':<18} | {'Delta':<10}")
        print("-" * 68)
        print(f"{'PR-AUC (Primary)':<20} | {base_m['pr_auc']:<18.4f} | {gbdt_m['pr_auc']:<18.4f} | {comp['pr_auc_delta']:<+10.4f}")
        print(f"{'ROC-AUC':<20} | {base_m['roc_auc']:<18.4f} | {gbdt_m['roc_auc']:<18.4f} | {comp['roc_auc_delta']:<+10.4f}")
        print(f"{'Precision (at 0.5)':<20} | {base_m['precision']:<18.4f} | {gbdt_m['precision']:<18.4f} | {gbdt_m['precision'] - base_m['precision']:<+10.4f}")
        print(f"{'Recall (at 0.5)':<20} | {base_m['recall']:<18.4f} | {gbdt_m['recall']:<18.4f} | {gbdt_m['recall'] - base_m['recall']:<+10.4f}")
        print(f"{'F1 Score':<20} | {base_m['f1_score']:<18.4f} | {gbdt_m['f1_score']:<18.4f} | {comp['f1_delta']:<+10.4f}")
        print("-" * 68)
        print(f"ℹ️  Note: Default 0.5 decision threshold is shifted due to scale_pos_weight.")
        print(f"   Precision/Recall trade-off will be calibrated across full threshold sweep in Layer 4.")

        print("\n🔍 TOP 5 FEATURES BY GAIN IMPORTANCE:")
        for rank, item in enumerate(results["top_features_by_gain"][:5], 1):
            print(f"  {rank}. {item['feature']:<35} (Gain: {item['gain_importance']:.2f})")

        print("\n📁 GENERATED ARTIFACTS:")
        print(f"• GBDT Model:      {results['artifact_paths']['gbdt_model']}")
        print(f"• Baseline Model:  {results['artifact_paths']['baseline_model']}")
        print(f"• Metrics Record:  {results['artifact_paths']['metrics']}")
        print("=" * 68 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
