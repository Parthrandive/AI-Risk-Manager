import os
import json
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_merge(
    transaction_path: str,
    identity_path: str,
    id_col: str = "TransactionID"
) -> pd.DataFrame:
    """
    Loads train_transaction.csv and train_identity.csv and merges them on TransactionID.
    Uses a left join with transaction as the base so all transactions are preserved.
    """
    if not os.path.exists(transaction_path):
        raise FileNotFoundError(f"Transaction file not found: {transaction_path}")
    
    logger.info(f"Loading transaction data from {transaction_path}...")
    df_trans = pd.read_csv(transaction_path)
    logger.info(f"Loaded {len(df_trans):,} transactions with {df_trans.shape[1]} columns.")

    if os.path.exists(identity_path):
        logger.info(f"Loading identity data from {identity_path}...")
        df_id = pd.read_csv(identity_path)
        logger.info(f"Loaded {len(df_id):,} identity records with {df_id.shape[1]} columns.")
        
        logger.info("Performing left join on %s...", id_col)
        merged_df = pd.merge(df_trans, df_id, on=id_col, how="left")
        logger.info(f"Merged dataset shape: {merged_df.shape}")
    else:
        logger.warning(
            f"Identity file {identity_path} not found. Proceeding with transaction data only."
        )
        merged_df = df_trans

    return merged_df


def sort_chronological(
    df: pd.DataFrame,
    time_col: str = "TransactionDT"
) -> pd.DataFrame:
    """
    Sorts dataframe strictly ascending by time column without random shuffling.
    """
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found in DataFrame.")
    
    logger.info(f"Sorting dataframe chronologically by '{time_col}'...")
    df_sorted = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
    return df_sorted


def compute_split_boundary(
    df: pd.DataFrame,
    quantile: float = 0.8,
    time_col: str = "TransactionDT"
) -> float:
    """
    Finds the time boundary at the specified quantile (e.g. 0.8 for 80/20 train/test split).
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")
    
    boundary = df[time_col].quantile(quantile)
    logger.info(
        f"Calculated {quantile*100:.1f}th percentile split boundary on '{time_col}': {boundary}"
    )
    return float(boundary)


def split_data(
    df: pd.DataFrame,
    boundary: float,
    time_col: str = "TransactionDT"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataframe into train (<= boundary) and test (> boundary).
    """
    train_df = df[df[time_col] <= boundary].copy().reset_index(drop=True)
    test_df = df[df[time_col] > boundary].copy().reset_index(drop=True)
    
    logger.info(f"Train split size: {len(train_df):,} rows ({len(train_df)/len(df)*100:.2f}%)")
    logger.info(f"Test split size:  {len(test_df):,} rows ({len(test_df)/len(df)*100:.2f}%)")
    return train_df, test_df


def compute_imbalance_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "isFraud"
) -> Dict[str, Any]:
    """
    Computes class imbalance (fraud rate) and summary metrics separately for train and test sets.
    """
    stats = {}
    for name, df in [("train", train_df), ("test", test_df)]:
        if target_col in df.columns:
            total_count = int(len(df))
            fraud_count = int(df[target_col].sum())
            legit_count = total_count - fraud_count
            fraud_rate = float(df[target_col].mean())
            stats[name] = {
                "total_count": total_count,
                "fraud_count": fraud_count,
                "legitimate_count": legit_count,
                "fraud_rate": fraud_rate,
                "fraud_percentage_str": f"{fraud_rate * 100:.3f}%"
            }
        else:
            stats[name] = {
                "total_count": int(len(df)),
                "note": f"Target column '{target_col}' not present"
            }

    if "train" in stats and "test" in stats and "fraud_rate" in stats["train"] and "fraud_rate" in stats["test"]:
        train_rate = stats["train"]["fraud_rate"]
        test_rate = stats["test"]["fraud_rate"]
        drift = test_rate - train_rate
        stats["drift"] = {
            "absolute_diff": float(drift),
            "relative_change_pct": float((drift / train_rate) * 100) if train_rate > 0 else 0.0,
            "drift_summary": f"Fraud rate shifted from {train_rate*100:.3f}% (train) to {test_rate*100:.3f}% (test)"
        }

    return stats


def verify_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    boundary: float,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT"
) -> Dict[str, bool]:
    """
    Verifies strict zero-leakage conditions:
    1. Zero TransactionID overlap between train and test splits.
    2. All train rows have TransactionDT <= boundary.
    3. All test rows have TransactionDT > boundary.
    """
    train_ids = set(train_df[id_col])
    test_ids = set(test_df[id_col])
    overlap_ids = train_ids.intersection(test_ids)
    
    no_id_overlap = (len(overlap_ids) == 0)
    train_time_valid = bool((train_df[time_col] <= boundary).all())
    test_time_valid = bool((test_df[time_col] > boundary).all())

    verification_results = {
        "no_id_overlap": no_id_overlap,
        "train_time_lte_boundary": train_time_valid,
        "test_time_gt_boundary": test_time_valid,
        "all_passed": (no_id_overlap and train_time_valid and test_time_valid)
    }

    if not no_id_overlap:
        logger.error(f"LEAKAGE DETECTED: {len(overlap_ids)} TransactionIDs exist in both train and test!")
    if not train_time_valid:
        max_train_t = train_df[time_col].max()
        logger.error(f"LEAKAGE DETECTED: Train split contains TransactionDT {max_train_t} > boundary {boundary}!")
    if not test_time_valid:
        min_test_t = test_df[time_col].min()
        logger.error(f"LEAKAGE DETECTED: Test split contains TransactionDT {min_test_t} <= boundary {boundary}!")

    if verification_results["all_passed"]:
        logger.info("✅ Zero-leakage verification passed: zero ID overlap and strict temporal boundaries.")
    else:
        raise ValueError(f"Leakage verification failed: {verification_results}")

    return verification_results


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed",
    metadata: Dict[str, Any] = None
) -> Tuple[str, str, str]:
    """
    Saves train and test splits to parquet files and records metadata.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    meta_path = os.path.join(output_dir, "split_metadata.json")

    logger.info(f"Saving train split to {train_path}...")
    train_df.to_parquet(train_path, index=False, engine="pyarrow", compression="snappy")

    logger.info(f"Saving test split to {test_path}...")
    test_df.to_parquet(test_path, index=False, engine="pyarrow", compression="snappy")

    if metadata:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved split metadata to {meta_path}.")

    return train_path, test_path, meta_path


def run_layer1_pipeline(
    transaction_path: str = "data/raw/train_transaction.csv",
    identity_path: str = "data/raw/train_identity.csv",
    output_dir: str = "data/processed",
    quantile: float = 0.8,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    target_col: str = "isFraud"
) -> Dict[str, Any]:
    """
    Orchestrates the entire Layer 1 data ingestion, chronological splitting,
    leakage verification, and parquet export pipeline.
    """
    logger.info("=== Starting Layer 1: Data & Chronological Split Pipeline ===")
    
    # 1. Load and merge
    df = load_and_merge(transaction_path, identity_path, id_col=id_col)
    
    # 2. Sort chronologically
    df_sorted = sort_chronological(df, time_col=time_col)
    
    # 3. Pick split boundary
    boundary = compute_split_boundary(df_sorted, quantile=quantile, time_col=time_col)
    
    # 4. Split data
    train_df, test_df = split_data(df_sorted, boundary=boundary, time_col=time_col)
    
    # 5. Class imbalance stats
    imbalance_stats = compute_imbalance_stats(train_df, test_df, target_col=target_col)
    logger.info(f"Class imbalance statistics: {json.dumps(imbalance_stats, indent=2)}")
    
    # 6. Verify zero leakage
    verification = verify_no_leakage(train_df, test_df, boundary, id_col=id_col, time_col=time_col)
    
    # 7. Save splits
    metadata = {
        "quantile": quantile,
        "boundary_transaction_dt": boundary,
        "total_records": len(df_sorted),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "imbalance_stats": imbalance_stats,
        "leakage_verification": verification,
        "raw_sources": {
            "transaction_path": transaction_path,
            "identity_path": identity_path
        }
    }
    
    train_p, test_p, meta_p = save_splits(train_df, test_df, output_dir=output_dir, metadata=metadata)
    
    logger.info("=== Layer 1 Complete: Output saved to %s ===", output_dir)
    return metadata
