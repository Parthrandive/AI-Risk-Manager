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
        "card2": 150,
        "card3": 150,
        "card4": np.random.choice(["visa", "mastercard"], size=n_train),
        "card5": 100,
        "card6": np.random.choice(["credit", "debit"], size=n_train),
        "addr1": np.random.choice([100, 200], size=n_train),
        "addr2": 87,
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "protonmail.com"], size=n_train),
        "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com"], size=n_train),
        "DeviceType": np.random.choice(["desktop", "mobile", np.nan], size=n_train),
        "DeviceInfo": np.random.choice(["iOS", "Windows", np.nan], size=n_train)
    })

    # Test data: timestamps 5100 to 10000 (strictly after train)
    t_test = np.sort(np.random.choice(np.arange(5100, 10000), size=n_test, replace=False))
    test_df = pd.DataFrame({
        "TransactionID": np.arange(2000, 2000 + n_test),
        "TransactionDT": t_test,
        "TransactionAmt": np.random.uniform(10, 300, size=n_test).astype(np.float32),
        "isFraud": np.random.choice([0, 1], size=n_test, p=[0.9, 0.1]),
        "ProductCD": np.random.choice(["W", "R", np.nan], size=n_test), # 'R' unseen
        "card1": np.random.choice([1001, 9999], size=n_test),
        "card2": 150,
        "card3": 150,
        "card4": np.random.choice(["visa", "discover"], size=n_test),
        "card5": 100,
        "card6": np.random.choice(["credit", "debit"], size=n_test),
        "addr1": np.random.choice([100, 300], size=n_test),
        "addr2": 87,
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
    assert pipeline.category_mappings["ProductCD"]["MISSING"] == 0
    assert pipeline.category_mappings["ProductCD"]["UNSEEN"] == 1


def test_transform_handles_unseen_and_missing(synthetic_splits):
    train_df, test_df = synthetic_splits
    pipeline = RiskFeaturePipeline().fit(train_df)

    transformed_train, transformed_test = pipeline.transform_splits(train_df, test_df)

    assert "ProductCD_encoded" in transformed_train.columns
    assert "ProductCD_encoded" in transformed_test.columns

    r_rows = test_df[test_df["ProductCD"] == "R"].index
    if len(r_rows) > 0:
        assert (transformed_test.loc[r_rows, "ProductCD_encoded"] == 1).all()


def test_expanding_card_mean_no_future_leakage():
    """Verifies that amt_to_expanding_card_mean_ratio uses strictly historical mean."""
    pipeline = RiskFeaturePipeline()
    # 3 sequential transactions for identical card: amounts 100, 200, 300
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [100, 200, 300],
        "TransactionAmt": [100.0, 200.0, 300.0],
        "card1": [1000, 1000, 1000],
        "card2": [1, 1, 1],
        "card3": [1, 1, 1],
        "card4": ["visa", "visa", "visa"],
        "card5": [1, 1, 1],
        "card6": ["credit", "credit", "credit"],
        "addr1": [10, 10, 10],
        "addr2": [10, 10, 10]
    })
    pipeline.fit(df)
    transformed = pipeline.transform(df)

    # Txn 1: first seen, uses initial fallback
    # Txn 2: historical mean = 100.0 -> ratio = 200 / 100 = 2.0
    # Txn 3: historical mean = (100+200)/2 = 150.0 -> ratio = 300 / 150 = 2.0
    ratios = transformed["amt_to_expanding_card_mean_ratio"].values
    assert np.isclose(ratios[1], 2.0)
    assert np.isclose(ratios[2], 2.0)


def test_state_continuity_across_split_boundary():
    """Verifies that test split retains velocity context from the end of train (no cold start)."""
    # Card 9999 has a transaction in train at t=4900
    train_df = pd.DataFrame({
        "TransactionID": [101],
        "TransactionDT": [4900],
        "TransactionAmt": [100.0],
        "card1": [9999],
        "card2": [1],
        "card3": [1],
        "card4": ["visa"],
        "card5": [1],
        "card6": ["credit"],
        "addr1": [10],
        "addr2": [10]
    })

    # Card 9999 has another transaction in test at t=5100 (200s later, within 10m window)
    test_df = pd.DataFrame({
        "TransactionID": [201],
        "TransactionDT": [5100],
        "TransactionAmt": [150.0],
        "card1": [9999],
        "card2": [1],
        "card3": [1],
        "card4": ["visa"],
        "card5": [1],
        "card6": ["credit"],
        "addr1": [10],
        "addr2": [10]
    })

    pipeline = RiskFeaturePipeline().fit(train_df)
    train_feat, test_feat = pipeline.transform_splits(train_df, test_df)

    # In test set:
    # 1. Recency should be 5100 - 4900 = 200s (NOT -1)
    assert test_feat.loc[0, "time_since_last_txn_card"] == 200.0
    # 2. 10m velocity should be 1 (NOT 0)
    assert test_feat.loc[0, "card_txn_count_10m"] == 1
    # 3. Expanding card mean ratio should be 150 / 100 = 1.5
    assert np.isclose(test_feat.loc[0, "amt_to_expanding_card_mean_ratio"], 1.5)


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

        loaded_pipeline = joblib.load(metadata["artifact_paths"]["pipeline"])
        assert loaded_pipeline.is_fitted
