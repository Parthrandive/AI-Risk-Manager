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
        description="Layer 3: Train baseline, XGBoost, and LightGBM models on held-out test split."
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
        help="Number of trees for GBDT"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max tree depth for GBDT"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for GBDT"
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=1.0,
        help="Positive class weight (1.0 for unweighted ranking optimization)"
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
            learning_rate=args.learning_rate,
            scale_pos_weight=args.scale_pos_weight
        )

        base_m = results["baseline_metrics"]
        xgb_m = results["xgboost_metrics"]
        lgb_m = results["lightgbm_metrics"]

        print("\n" + "=" * 80)
        print("🚀 LAYER 3: MODEL BENCHMARKING & COMPARISON SUMMARY")
        print("=" * 80)
        print(f"• Evaluated Features:  {results['feature_count']}")
        print(f"• Scale Pos Weight:    {xgb_m.get('scale_pos_weight_used', 1.0):.2f}")
        
        print("\n📊 HELD-OUT TEST SET BENCHMARK RESULTS:")
        print(f"{'Metric':<22} | {'Baseline (LogReg)':<18} | {'XGBoost GBDT':<18} | {'LightGBM GBDT':<18}")
        print("-" * 80)
        print(f"{'PR-AUC (Primary)':<22} | {base_m['pr_auc']:<18.4f} | {xgb_m['pr_auc']:<18.4f} | {lgb_m['pr_auc']:<18.4f}")
        print(f"{'ROC-AUC':<22} | {base_m['roc_auc']:<18.4f} | {xgb_m['roc_auc']:<18.4f} | {lgb_m['roc_auc']:<18.4f}")
        print(f"{'Precision (at 0.5)':<22} | {base_m['precision']:<18.4f} | {xgb_m['precision']:<18.4f} | {lgb_m['precision']:<18.4f}")
        print(f"{'Recall (at 0.5)':<22} | {base_m['recall']:<18.4f} | {xgb_m['recall']:<18.4f} | {lgb_m['recall']:<18.4f}")
        print(f"{'F1 Score':<22} | {base_m['f1_score']:<18.4f} | {xgb_m['f1_score']:<18.4f} | {lgb_m['f1_score']:<18.4f}")
        print("-" * 80)
        print(f"ℹ️  Note on Precision/Recall: Default 0.5 cutoff is shifted due to scale_pos_weight.")
        print(f"   The calibrated threshold sweep & false-positive cost analysis follow in Layer 4.")

        print("\n🔍 TOP 5 FEATURES BY GAIN IMPORTANCE (XGBoost):")
        for rank, item in enumerate(results["top_features_by_gain"][:5], 1):
            print(f"  {rank}. {item['feature']:<35} (Gain: {item['gain_importance']:.2f})")

        print("\n📁 GENERATED ARTIFACTS:")
        print(f"• XGBoost Model:   {results['artifact_paths']['xgboost_model']}")
        print(f"• LightGBM Model:  {results['artifact_paths']['lightgbm_model']}")
        print(f"• Baseline Model:  {results['artifact_paths']['baseline_model']}")
        print(f"• Metrics Record:  {results['artifact_paths']['metrics']}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
