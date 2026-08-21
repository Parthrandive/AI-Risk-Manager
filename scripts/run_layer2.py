#!/usr/bin/env python3
"""
CLI Script to execute Layer 2 (Feature Engineering & Preprocessing) of the AI Risk Manager pipeline.
"""

import argparse
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import run_layer2_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Layer 2: Fit leak-free preprocessors on train only, compute forward velocities, and engineer risk features."
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default="data/processed/train.parquet",
        help="Path to train.parquet (from Layer 1)"
    )
    parser.add_argument(
        "--test-parquet",
        type=str,
        default="data/processed/test.parquet",
        help="Path to test.parquet (from Layer 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save train_features.parquet, test_features.parquet, and pipeline artifacts"
    )

    args = parser.parse_args()

    if not os.path.exists(args.train_parquet) or not os.path.exists(args.test_parquet):
        print("\n❌ Error: Layer 1 output files not found.")
        print(f"Missing: {args.train_parquet} or {args.test_parquet}")
        print("Please run Layer 1 first: python3 scripts/run_layer1.py")
        sys.exit(1)

    try:
        metadata = run_layer2_pipeline(
            train_parquet_path=args.train_parquet,
            test_parquet_path=args.test_parquet,
            output_dir=args.output_dir
        )
        print("\n" + "=" * 60)
        print("🚀 LAYER 2 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"• Transformed Train Records: {metadata['train_rows']:,}")
        print(f"• Transformed Test Records:  {metadata['test_rows']:,}")
        print(f"• Model-Ready Features:      {metadata['feature_count']}")
        print(f"• Categoricals Encoded:      {len(metadata['categorical_columns_encoded'])}")
        print(f"• Global Amount Median:      ${metadata['global_amt_median']:.2f}")
        print("\n📁 GENERATED ARTIFACTS:")
        print(f"• Train Features:  {metadata['artifact_paths']['train_features']}")
        print(f"• Test Features:   {metadata['artifact_paths']['test_features']}")
        print(f"• Fitted Pipeline: {metadata['artifact_paths']['pipeline']}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
