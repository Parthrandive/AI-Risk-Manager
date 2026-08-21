import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import RiskFeaturePipeline, run_layer2_pipeline


@pytest.fixture
def synthetic_splits():
    """Generates synthetic train and test splits."""
    n_train, n_test = 60, 20
    
    # Train data: timestamps 100 to 5000
    t_train = np.sort(np.random.choice(np.arange(100, 5000), size=n_train, replace=False))
    train_df = pd.DataFrame({
        "TransactionID": np.arange(1000, 1000 + n_train),
        "TransactionDT": t_train,
        "TransactionAmt": np.random.uniform(10, 300, size=n_train).astype(np.float32),
        "isFraud": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "ProductCD": np.random.choice(["W", "C", "H", np.nan], size=n_train),
        "card1": np.random.choice([1001, 1002, 1003], size=n_train),
        "card4": np.random.choice(["visa", "mastercard"], size=n_train),
        "card6": np.random.choice(["credit", "debit"], size=n_train),
        "addr1": np.random.choice([100, 200], size=n_train),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "protonmail.com"], size=n_train),
        "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com"], size=n_train),
        "DeviceType": np.random.choice(["desktop", "mobile", np.nan], size=n_train),
        "DeviceInfo": np.random.choice(["iOS", "Windows", np.nan], size=n_train)
    })

    # Test data: timestamps 5100 to 10000 (strictly after train)
    # Includes a new/unseen ProductCD 'R' and card4 'discover'
    t_test = np.sort(np.random.choice(np.arange(5100, 10000), size=n_test, replace=False))
    test_df = pd.DataFrame({
        "TransactionID": np.arange(2000, 2000 + n_test),
        "TransactionDT": t_test,
        "TransactionAmt": np.random.uniform(10, 300, size=n_test).astype(np.float32),
        "isFraud": np.random.choice([0, 1], size=n_test, p=[0.9, 0.1]),
        "ProductCD": np.random.choice(["W", "R", np.nan], size=n_test), # 'R' is unseen in train
        "card1": np.random.choice([1001, 9999], size=n_test),           # 9999 unseen in train
        "card4": np.random.choice(["visa", "discover"], size=n_test),    # 'discover' unseen
        "card6": np.random.choice(["credit", "debit"], size=n_test),
        "addr1": np.random.choice([100, 300], size=n_test),
        "P_emaildomain": np.random.choice(["gmail.com", "anonymous.com"], size=n_test),
        "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com"], size=n_test),
        "DeviceType": np.random.choice(["desktop", "mobile", np.nan], size=n_test),
        "DeviceInfo": np.random.choice(["iOS", "Android"], size=n_test)
    })

    return train_df, test_df


def test_fit_on_train_only(synthetic_splits):
    train_df, _ = synthetic_splits
    pipeline = RiskFeaturePipeline()
    assert not pipeline.is_fitted
    
    pipeline.fit(train_df)
    assert pipeline.is_fitted
    assert "ProductCD" in pipeline.category_mappings
    # Unseen code is 1, Missing is 0
    assert pipeline.category_mappings["ProductCD"]["MISSING"] == 0
    assert pipeline.category_mappings["ProductCD"]["UNSEEN"] == 1


def test_transform_handles_unseen_and_missing(synthetic_splits):
    train_df, test_df = synthetic_splits
    pipeline = RiskFeaturePipeline().fit(train_df)

    transformed_train = pipeline.transform(train_df)
    transformed_test = pipeline.transform(test_df)

    # Check encoded column exists
    assert "ProductCD_encoded" in transformed_train.columns
    assert "ProductCD_encoded" in transformed_test.columns

    # Test set contains unseen category 'R' which must be mapped to 1 (UNSEEN)
    r_rows = test_df[test_df["ProductCD"] == "R"].index
    if len(r_rows) > 0:
        assert (transformed_test.loc[r_rows, "ProductCD_encoded"] == 1).all()


def test_forward_velocities(synthetic_splits):
    train_df, _ = synthetic_splits
    pipeline = RiskFeaturePipeline().fit(train_df)
    transformed = pipeline.transform(train_df)

    # Check velocity columns
    assert "time_since_last_txn_card" in transformed.columns
    assert "card_txn_count_10m" in transformed.columns
    assert "card_txn_count_1h" in transformed.columns
    assert "card_txn_count_24h" in transformed.columns

    # Velocity counts must be >= 0
    assert (transformed["card_txn_count_10m"] >= 0).all()
    assert (transformed["card_txn_count_1h"] >= transformed["card_txn_count_10m"]).all()
    assert (transformed["card_txn_count_24h"] >= transformed["card_txn_count_1h"]).all()


def test_amount_features(synthetic_splits):
    train_df, _ = synthetic_splits
    pipeline = RiskFeaturePipeline().fit(train_df)
    transformed = pipeline.transform(train_df)

    assert "TransactionAmt_log" in transformed.columns
    assert "TransactionAmt_decimal" in transformed.columns
    assert "amt_to_card_mean_ratio" in transformed.columns
    assert (transformed["amt_to_card_mean_ratio"] > 0).all()


def test_run_layer2_pipeline_end_to_end(synthetic_splits):
    train_df, test_df = synthetic_splits
    with tempfile.TemporaryDirectory() as tmpdir:
        train_p = os.path.join(tmpdir, "train.parquet")
        test_p = os.path.join(tmpdir, "test.parquet")
        out_dir = os.path.join(tmpdir, "processed")

        train_df.to_parquet(train_p, index=False)
        test_df.to_parquet(test_p, index=False)

        metadata = run_layer2_pipeline(
            train_parquet_path=train_p,
            test_parquet_path=test_p,
            output_dir=out_dir
        )

        assert os.path.exists(metadata["artifact_paths"]["train_features"])
        assert os.path.exists(metadata["artifact_paths"]["test_features"])
        assert os.path.exists(metadata["artifact_paths"]["pipeline"])

        # Check deserialization of fitted pipeline
        loaded_pipeline = joblib.load(metadata["artifact_paths"]["pipeline"])
        assert loaded_pipeline.is_fitted
