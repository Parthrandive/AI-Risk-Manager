import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.walk_forward import (
    partition_into_periods,
    evaluate_predictions,
    run_walk_forward_experiment
)


def test_partition_into_periods():
    n = 500
    df = pd.DataFrame({
        "TransactionID": np.arange(1000, 1000 + n),
        "TransactionDT": np.linspace(1000, 10000, n),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "TransactionAmt": np.random.uniform(10, 100, size=n)
    })

    part_df, meta = partition_into_periods(df, n_periods=5)

    assert "period" in part_df.columns
    assert set(part_df["period"].unique()) == {1, 2, 3, 4, 5}
    assert len(meta) == 5

    # Check non-overlapping periods
    for p in range(1, 5):
        assert meta[p]["dt_max"] <= meta[p + 1]["dt_min"]


def test_evaluate_predictions():
    np.random.seed(42)
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_true[:20] = 1 # 10% fraud

    y_prob = np.where(y_true == 1, np.random.uniform(0.6, 0.9, n), np.random.uniform(0.01, 0.4, n))

    metrics = evaluate_predictions(y_true, y_prob)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert "calibrated_3pct_policy" in metrics
    assert "recall" in metrics["calibrated_3pct_policy"]


def test_run_walk_forward_experiment_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(42)
        n = 300
        y = np.zeros(n, dtype=int)
        y[:20] = 1

        df = pd.DataFrame({
            "TransactionID": np.arange(1000, 1000 + n),
            "TransactionDT": np.linspace(1000, 50000, n),
            "isFraud": y,
            "TransactionAmt": np.random.uniform(10, 200, size=n).astype(np.float32),
            "card1": np.random.choice([1001, 1002], size=n),
            "card2": 150,
            "card3": 150,
            "card4": "visa",
            "card5": 100,
            "card6": "credit",
            "addr1": 100,
            "addr2": 87,
            "ProductCD": "W",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com"
        })

        train_df = df.iloc[:200]
        test_df = df.iloc[200:]

        train_p = os.path.join(tmpdir, "train.parquet")
        test_p = os.path.join(tmpdir, "test.parquet")
        train_df.to_parquet(train_p, index=False)
        test_df.to_parquet(test_p, index=False)

        results = run_walk_forward_experiment(
            train_parquet_path=train_p,
            test_parquet_path=test_p,
            output_dir=tmpdir,
            n_periods=5
        )

        assert "period_partition_metadata" in results
        assert "frozen_model_evaluation" in results
        assert "rolling_retrain_evaluation" in results
        assert "drift_summary" in results
