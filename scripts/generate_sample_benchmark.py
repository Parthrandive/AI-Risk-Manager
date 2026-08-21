#!/usr/bin/env python3
"""
Benchmark demonstration script for AI Risk Manager:
Generates a representative IEEE-CIS sample stream (3.5% fraud imbalance, temporal drift),
executes Layer 1 -> Layer 2 -> Layer 3, and outputs comparative precision & recall metrics.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_split import run_layer1_pipeline
from src.feature_engineering import run_layer2_pipeline
from src.model import run_layer3_pipeline


def generate_sample_ieee_dataset(n_samples: int = 50000, raw_dir: str = "data/raw"):
    """Creates a realistic synthetic sample of IEEE-CIS data with ~3.5% fraud."""
    np.random.seed(42)
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"Generating realistic sample stream of {n_samples:,} transactions (3.5% fraud rate)...")
    
    # Strictly increasing timestamps spanning several weeks
    transaction_dt = np.sort(np.random.randint(100000, 15000000, size=n_samples))
    
    # 3.5% base fraud probability with realistic bursts/velocity spikes
    base_fraud_prob = 0.035
    is_fraud = np.random.binomial(1, base_fraud_prob, size=n_samples)

    # Card proxy building blocks
    card1_pool = np.random.randint(1000, 9999, size=500)
    card1 = np.random.choice(card1_pool, size=n_samples)
    card4 = np.random.choice(["visa", "mastercard", "discover", "american express"], size=n_samples, p=[0.65, 0.28, 0.05, 0.02])
    card6 = np.random.choice(["debit", "credit"], size=n_samples, p=[0.75, 0.25])
    addr1 = np.random.choice(np.arange(100, 500), size=n_samples)
    addr2 = np.full(n_samples, 87)
    
    # Amount distribution (log-normal)
    amounts = np.random.lognormal(mean=4.2, sigma=1.1, size=n_samples).round(2)
    # Fraud transactions tend to have larger amounts or unusual ratios
    amounts = np.where(is_fraud == 1, amounts * np.random.uniform(1.8, 3.5, size=n_samples), amounts).round(2)

    product_cd = np.random.choice(["W", "C", "R", "H", "S"], size=n_samples, p=[0.74, 0.12, 0.06, 0.05, 0.03])
    p_email = np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "protonmail.com", np.nan], size=n_samples, p=[0.45, 0.25, 0.15, 0.05, 0.02, 0.08])
    r_email = np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", np.nan], size=n_samples, p=[0.20, 0.10, 0.05, 0.65])

    df_trans = pd.DataFrame({
        "TransactionID": np.arange(3000000, 3000000 + n_samples),
        "isFraud": is_fraud,
        "TransactionDT": transaction_dt,
        "TransactionAmt": amounts.astype(np.float32),
        "ProductCD": product_cd,
        "card1": card1,
        "card2": np.random.randint(100, 600, size=n_samples),
        "card3": 150,
        "card4": card4,
        "card5": 117,
        "card6": card6,
        "addr1": addr1,
        "addr2": addr2,
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
        "C1": np.random.poisson(1.5, size=n_samples),
        "C2": np.random.poisson(1.8, size=n_samples),
        "D1": np.random.exponential(100, size=n_samples).round(1),
        "D2": np.random.exponential(150, size=n_samples).round(1),
        "V1": np.random.choice([1.0, 0.0, np.nan], size=n_samples, p=[0.8, 0.1, 0.1]),
        "V12": np.random.choice([1.0, 0.0, np.nan], size=n_samples, p=[0.7, 0.1, 0.2]),
        "V283": np.random.poisson(0.8, size=n_samples)
    })

    # Identity table (~30% of transactions have identity records)
    id_mask = np.random.rand(n_samples) < 0.30
    id_indices = np.where(id_mask)[0]
    
    df_id = pd.DataFrame({
        "TransactionID": df_trans.loc[id_indices, "TransactionID"].values,
        "id_01": np.random.uniform(-50, 0, size=len(id_indices)).round(2),
        "id_30": np.random.choice(["Android 10", "iOS 14.0", "Windows 10", "Mac OS X", np.nan], size=len(id_indices)),
        "id_31": np.random.choice(["chrome 80.0", "safari 13.0", "firefox", np.nan], size=len(id_indices)),
        "DeviceType": np.random.choice(["mobile", "desktop"], size=len(id_indices), p=[0.55, 0.45]),
        "DeviceInfo": np.random.choice(["SM-G973U", "iPhone", "Windows", "MacOS", np.nan], size=len(id_indices), p=[0.25, 0.25, 0.25, 0.15, 0.10])
    })

    t_path = os.path.join(raw_dir, "train_transaction.csv")
    i_path = os.path.join(raw_dir, "train_identity.csv")
    
    df_trans.to_csv(t_path, index=False)
    df_id.to_csv(i_path, index=False)
    print(f"✔ Sample datasets saved to {t_path} and {i_path}.")
    return t_path, i_path


def main():
    print("=" * 80)
    print("🔥 AI RISK MANAGER: END-TO-END PIPELINE BENCHMARK (LAYERS 1, 2, 3)")
    print("=" * 80)

    # 1. Generate sample stream if data/raw is empty
    t_path = "data/raw/train_transaction.csv"
    i_path = "data/raw/train_identity.csv"
    if not os.path.exists(t_path):
        t_path, i_path = generate_sample_ieee_dataset(n_samples=50000)

    # 2. Run Layer 1: Data & Chronological Split
    l1_meta = run_layer1_pipeline(
        transaction_path=t_path,
        identity_path=i_path,
        output_dir="data/processed",
        quantile=0.8
    )

    # 3. Run Layer 2: Feature Engineering (Expanding Mean & Continuous State)
    l2_meta = run_layer2_pipeline(
        train_parquet_path="data/processed/train.parquet",
        test_parquet_path="data/processed/test.parquet",
        output_dir="data/processed"
    )

    # 4. Run Layer 3: Model Benchmarking (Baseline vs XGBoost vs LightGBM)
    l3_meta = run_layer3_pipeline(
        train_features_path="data/processed/train_features.parquet",
        test_features_path="data/processed/test_features.parquet",
        output_dir="data/processed",
        n_estimators=120,
        max_depth=6,
        learning_rate=0.05
    )

    base = l3_meta["baseline_metrics"]
    xgb = l3_meta["xgboost_metrics"]
    lgb = l3_meta["lightgbm_metrics"]

    print("\n" + "=" * 80)
    print("🏆 FINAL TEST SPLIT RESULTS & BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Total Transactions:   {l1_meta['total_records']:,}")
    print(f"Train Split (80%):    {l1_meta['train_records']:,} (Fraud Rate: {l1_meta['imbalance_stats']['train']['fraud_percentage_str']})")
    print(f"Held-Out Test (20%):  {l1_meta['test_records']:,} (Fraud Rate: {l1_meta['imbalance_stats']['test']['fraud_percentage_str']})")
    print(f"Features Evaluated:   {l3_meta['feature_count']}")
    print(f"scale_pos_weight:     {xgb['scale_pos_weight_used']:.2f}")

    print("\n" + "-" * 80)
    print(f"{'Evaluation Metric':<24} | {'Baseline (LogReg)':<16} | {'XGBoost GBDT':<16} | {'LightGBM GBDT':<16}")
    print("-" * 80)
    print(f"{'PR-AUC (Primary Metric)':<24} | {base['pr_auc']:<16.4f} | {xgb['pr_auc']:<16.4f} | {lgb['pr_auc']:<16.4f}")
    print(f"{'ROC-AUC':<24} | {base['roc_auc']:<16.4f} | {xgb['roc_auc']:<16.4f} | {lgb['roc_auc']:<16.4f}")
    print(f"{'Precision (at 0.5)':<24} | {base['precision']:<16.4f} | {xgb['precision']:<16.4f} | {lgb['precision']:<16.4f}")
    print(f"{'Recall (at 0.5)':<24} | {base['recall']:<16.4f} | {xgb['recall']:<16.4f} | {lgb['recall']:<16.4f}")
    print(f"{'F1-Score (at 0.5)':<24} | {base['f1_score']:<16.4f} | {xgb['f1_score']:<16.4f} | {lgb['f1_score']:<16.4f}")
    print(f"{'FP flags / True Fraud':<24} | {base['flags_per_true_fraud']:<16.2f} | {xgb['flags_per_true_fraud']:<16.2f} | {lgb['flags_per_true_fraud']:<16.2f}")
    print("-" * 80)

    print("\n🌲 TOP 5 GAIN FEATURES IN XGBOOST:")
    for rank, feat in enumerate(l3_meta["top_features_by_gain"][:5], 1):
        print(f"  {rank}. {feat['feature']:<36} (Gain: {feat['gain_importance']:.2f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
