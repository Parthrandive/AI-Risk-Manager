import os
import sys
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_split import (
    load_and_merge,
    reduce_memory_usage,
    perform_sanity_checks,
    sort_chronological,
    compute_split_boundary,
    split_data,
    compute_imbalance_stats,
    verify_no_leakage,
    save_splits,
    run_layer1_pipeline
)


@pytest.fixture
def sample_data():
    """Creates synthetic transaction and identity DataFrames for testing."""
    n = 100
    transaction_ids = np.arange(1000, 1000 + n)
    transaction_dt = np.sort(np.random.choice(np.arange(100, 10000), size=n, replace=False))
    shuffled_dt = transaction_dt.copy()
    np.random.shuffle(shuffled_dt)
    
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[:5] = 1
    np.random.shuffle(is_fraud)

    df_trans = pd.DataFrame({
        "TransactionID": transaction_ids,
        "TransactionDT": shuffled_dt,
        "TransactionAmt": np.random.uniform(10, 500, size=n).astype(np.float64),
        "isFraud": is_fraud,
        "card1": np.random.randint(1000, 9999, size=n, dtype=np.int64)
    })

    df_id = pd.DataFrame({
        "TransactionID": transaction_ids[:50],
        "id_01": np.random.uniform(-10, 0, size=50),
        "DeviceType": ["mobile" if i % 2 == 0 else "desktop" for i in range(50)]
    })

    return df_trans, df_id


def test_sanity_checks_pass(sample_data):
    df_trans, _ = sample_data
    cleaned = perform_sanity_checks(df_trans)
    assert len(cleaned) == len(df_trans)


def test_sanity_checks_negative_amt_raises():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [100, 200],
        "TransactionAmt": [50.0, -10.0],
        "isFraud": [0, 1]
    })
    with pytest.raises(ValueError, match="negative 'TransactionAmt'"):
        perform_sanity_checks(df)


def test_sanity_checks_duplicate_id_raises():
    df = pd.DataFrame({
        "TransactionID": [1, 1],
        "TransactionDT": [100, 200],
        "TransactionAmt": [50.0, 10.0],
        "isFraud": [0, 1]
    })
    with pytest.raises(ValueError, match="duplicate 'TransactionID'"):
        perform_sanity_checks(df)


def test_reduce_memory_usage(sample_data):
    df_trans, _ = sample_data
    assert df_trans["TransactionAmt"].dtype == np.float64
    downcasted = reduce_memory_usage(df_trans)
    assert downcasted["TransactionAmt"].dtype == np.float32


def test_merge_and_sort(sample_data):
    df_trans, df_id = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        t_path = os.path.join(tmpdir, "trans.csv")
        i_path = os.path.join(tmpdir, "id.csv")
        df_trans.to_csv(t_path, index=False)
        df_id.to_csv(i_path, index=False)

        merged = load_and_merge(t_path, i_path)
        assert len(merged) == len(df_trans)
        assert "DeviceType" in merged.columns
        assert merged["DeviceType"].isna().sum() == 50

        sorted_df = sort_chronological(merged, "TransactionDT")
        assert sorted_df["TransactionDT"].is_monotonic_increasing


def test_split_boundary_and_partition(sample_data):
    df_trans, _ = sample_data
    sorted_df = sort_chronological(df_trans, "TransactionDT")
    
    boundary = compute_split_boundary(sorted_df, quantile=0.8)
    train_df, test_df = split_data(sorted_df, boundary=boundary)

    assert len(train_df) + len(test_df) == len(sorted_df)
    assert len(train_df) == int(len(sorted_df) * 0.8)
    assert len(test_df) == int(len(sorted_df) * 0.2)

    verification = verify_no_leakage(train_df, test_df, boundary)
    assert verification["all_passed"] is True
    assert verification["no_id_overlap"] is True
    assert verification["train_time_lte_boundary"] is True
    assert verification["test_time_gt_boundary"] is True


def test_leakage_detection_raises_error(sample_data):
    df_trans, _ = sample_data
    sorted_df = sort_chronological(df_trans, "TransactionDT")
    boundary = compute_split_boundary(sorted_df, quantile=0.8)
    train_df, test_df = split_data(sorted_df, boundary=boundary)

    corrupted_test = test_df.copy()
    corrupted_test.loc[0, "TransactionID"] = train_df.loc[0, "TransactionID"]
    corrupted_test.loc[0, "TransactionDT"] = train_df.loc[0, "TransactionDT"]

    with pytest.raises(ValueError, match="Leakage verification failed"):
        verify_no_leakage(train_df, corrupted_test, boundary)


def test_imbalance_stats(sample_data):
    df_trans, _ = sample_data
    sorted_df = sort_chronological(df_trans, "TransactionDT")
    boundary = compute_split_boundary(sorted_df, quantile=0.8)
    train_df, test_df = split_data(sorted_df, boundary=boundary)

    stats = compute_imbalance_stats(train_df, test_df)
    assert "train" in stats
    assert "test" in stats
    assert "fraud_rate" in stats["train"]
    assert "fraud_rate" in stats["test"]
    assert "drift" in stats


def test_end_to_end_pipeline(sample_data):
    df_trans, df_id = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        t_path = os.path.join(tmpdir, "train_transaction.csv")
        i_path = os.path.join(tmpdir, "train_identity.csv")
        out_dir = os.path.join(tmpdir, "processed")

        df_trans.to_csv(t_path, index=False)
        df_id.to_csv(i_path, index=False)

        metadata = run_layer1_pipeline(
            transaction_path=t_path,
            identity_path=i_path,
            output_dir=out_dir,
            quantile=0.8
        )

        assert os.path.exists(os.path.join(out_dir, "train.parquet"))
        assert os.path.exists(os.path.join(out_dir, "test.parquet"))
        assert os.path.exists(os.path.join(out_dir, "split_metadata.json"))
        assert metadata["leakage_verification"]["all_passed"] is True

        loaded_train = pd.read_parquet(os.path.join(out_dir, "train.parquet"))
        loaded_test = pd.read_parquet(os.path.join(out_dir, "test.parquet"))
        assert len(loaded_train) == metadata["train_records"]
        assert len(loaded_test) == metadata["test_records"]
