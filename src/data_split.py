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


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcasts numeric dtypes (float64 -> float32, int64 -> int32/int16/int8)
    to significantly reduce memory footprint before downstream processing.
    Safe for Layer 1 as it operates row-wise without dataset-wide statistical aggregation.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(col_type):
                # Using float32 for floating point numbers
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = (1 - end_mem / start_mem) * 100 if start_mem > 0 else 0
        logger.info(
            f"Memory downcasting: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction)"
        )
    return df


def perform_sanity_checks(
    df: pd.DataFrame,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    amt_col: str = "TransactionAmt",
    target_col: str = "isFraud"
) -> pd.DataFrame:
    """
    Executes leak-free data sanity checks:
    1. Removes exact duplicate rows if any exist.
    2. Verifies unique TransactionID.
    3. Verifies non-negative TransactionAmt.
    4. Verifies no nulls in time column (TransactionDT).
    5. Verifies target (isFraud) is strictly 0/1 without nulls.
    """
    logger.info("Running Layer 1 data cleanliness and sanity checks...")
    
    # 1. Exact duplicate rows
    init_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped_dups = init_len - len(df)
    if dropped_dups > 0:
        logger.warning(f"Dropped {dropped_dups} exact duplicate rows.")

    # 2. TransactionID uniqueness
    if id_col in df.columns:
        duplicate_ids = df[id_col].duplicated().sum()
        if duplicate_ids > 0:
            raise ValueError(f"Found {duplicate_ids} duplicate '{id_col}' values.")
        logger.info(f"✔ TransactionID uniqueness verified (0 duplicates).")

    # 3. Non-negative TransactionAmt
    if amt_col in df.columns:
        negative_amt = (df[amt_col] < 0).sum()
        if negative_amt > 0:
            raise ValueError(f"Found {negative_amt} negative '{amt_col}' values.")
        logger.info(f"✔ Transaction amount validity verified (all {amt_col} >= 0).")

    # 4. Monotonic-safe TransactionDT (no nulls)
    if time_col in df.columns:
        null_time = df[time_col].isna().sum()
        if null_time > 0:
            raise ValueError(f"Found {null_time} null values in time column '{time_col}'.")
        logger.info(f"✔ Time column '{time_col}' verified (0 nulls).")

    # 5. Strict 0/1 isFraud
    if target_col in df.columns:
        null_target = df[target_col].isna().sum()
        if null_target > 0:
            raise ValueError(f"Found {null_target} null values in target column '{target_col}'.")
        
        unique_targets = set(df[target_col].unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(f"Invalid target values in '{target_col}': {unique_targets}. Must be {{0, 1}}.")
        logger.info(f"✔ Target column '{target_col}' verified (strictly binary {{0, 1}}, 0 nulls).")

    return df


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

    if len(df_trans) >= 590000:
        logger.info("✔ Verified authentic full IEEE-CIS dataset (~590,540 transactions).")
    else:
        logger.warning(
            f"Dataset contains {len(df_trans):,} rows. Note: Full Kaggle IEEE-CIS train_transaction.csv has 590,540 rows."
        )

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
    amt_col: str = "TransactionAmt",
    target_col: str = "isFraud",
    downcast_memory: bool = True
) -> Dict[str, Any]:
    """
    Orchestrates the complete Layer 1 pipeline:
    1. Load and merge raw transactions & identity
    2. Data sanity checks (drop duplicates, assert no negative amount, assert valid target)
    3. Memory downcasting (float64 -> float32, int64 -> int32/int16/int8)
    4. Chronological sorting
    5. Time-quantile boundary calculation
    6. Non-shuffled train/test partition
    7. Class imbalance calculation & drift analysis
    8. Zero-leakage verification
    9. Parquet export
    """
    logger.info("=== Starting Layer 1: Data & Chronological Split Pipeline ===")
    
    # 1. Load and merge
    df = load_and_merge(transaction_path, identity_path, id_col=id_col)
    
    # 2. Sanity checks (cleanliness before split)
    df = perform_sanity_checks(df, id_col=id_col, time_col=time_col, amt_col=amt_col, target_col=target_col)

    # 3. Downcast memory
    if downcast_memory:
        df = reduce_memory_usage(df)

    # 4. Sort chronologically
    df_sorted = sort_chronological(df, time_col=time_col)
    
    # 5. Pick split boundary
    boundary = compute_split_boundary(df_sorted, quantile=quantile, time_col=time_col)
    
    # 6. Split data
    train_df, test_df = split_data(df_sorted, boundary=boundary, time_col=time_col)
    
    # 7. Class imbalance stats
    imbalance_stats = compute_imbalance_stats(train_df, test_df, target_col=target_col)
    logger.info(f"Class imbalance statistics: {json.dumps(imbalance_stats, indent=2)}")
    
    # 8. Verify zero leakage
    verification = verify_no_leakage(train_df, test_df, boundary, id_col=id_col, time_col=time_col)
    
    # 9. Save splits
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
