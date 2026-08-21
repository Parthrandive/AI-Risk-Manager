#!/usr/bin/env python3
"""
CLI Script to execute Layer 1 (Data & Chronological Split) of the AI Risk Manager pipeline.
"""

import argparse
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_split import run_layer1_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: Ingest raw IEEE-CIS fraud data, sort chronologically, and execute leak-free time-quantile split."
    )
    parser.add_argument(
        "--transaction-path",
        type=str,
        default="data/raw/train_transaction.csv",
        help="Path to train_transaction.csv"
    )
    parser.add_argument(
        "--identity-path",
        type=str,
        default="data/raw/train_identity.csv",
        help="Path to train_identity.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save train.parquet, test.parquet, and metadata"
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.8,
        help="Chronological train split quantile (e.g. 0.8 for 80%% train, 20%% test)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.transaction_path):
        print(f"\n❌ Error: '{args.transaction_path}' does not exist.")
        print("Please place 'train_transaction.csv' (and optional 'train_identity.csv') inside 'data/raw/'.")
        print("\nTo download from Kaggle:")
        print("  kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw/")
        print("  kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/raw/")
        sys.exit(1)

    try:
        metadata = run_layer1_pipeline(
            transaction_path=args.transaction_path,
            identity_path=args.identity_path,
            output_dir=args.output_dir,
            quantile=args.quantile
        )
        print("\n" + "=" * 60)
        print("🚀 LAYER 1 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"• Total Transactions Processed: {metadata['total_records']:,}")
        print(f"• Train Records (<= {metadata['boundary_transaction_dt']:.0f}s): {metadata['train_records']:,}")
        print(f"• Test Records  (>  {metadata['boundary_transaction_dt']:.0f}s): {metadata['test_records']:,}")
        
        train_stats = metadata["imbalance_stats"].get("train", {})
        test_stats = metadata["imbalance_stats"].get("test", {})
        drift_stats = metadata["imbalance_stats"].get("drift", {})

        print("\n📊 CLASS IMBALANCE (isFraud):")
        print(f"• Train Fraud Rate: {train_stats.get('fraud_percentage_str', 'N/A')} ({train_stats.get('fraud_count', 0):,} / {train_stats.get('total_count', 0):,})")
        print(f"• Test Fraud Rate:  {test_stats.get('fraud_percentage_str', 'N/A')} ({test_stats.get('fraud_count', 0):,} / {test_stats.get('total_count', 0):,})")
        if drift_stats:
            print(f"• Temporal Drift:   {drift_stats.get('drift_summary')}")

        print("\n🛡️ ZERO-LEAKAGE VERIFICATION:")
        leakage = metadata["leakage_verification"]
        print(f"• Zero ID Overlap:             {'✅ PASSED' if leakage['no_id_overlap'] else '❌ FAILED'}")
        print(f"• Train Timestamps <= Split:   {'✅ PASSED' if leakage['train_time_lte_boundary'] else '❌ FAILED'}")
        print(f"• Test Timestamps > Split:     {'✅ PASSED' if leakage['test_time_gt_boundary'] else '❌ FAILED'}")
        print(f"• Overall Status:              {'✅ 100% LEAK-FREE' if leakage['all_passed'] else '❌ LEAKAGE DETECTED'}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
