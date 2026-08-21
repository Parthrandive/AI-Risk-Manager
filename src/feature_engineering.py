import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Default high-importance categorical columns in IEEE-CIS
DEFAULT_CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
    "DeviceInfo"
]

# Default identity proxy components
CARD_ID_COLS = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1"]
DEVICE_ID_COLS = ["DeviceType", "DeviceInfo", "id_30", "id_31"]


class RiskFeaturePipeline:
    """
    Leak-Free Feature Engineering & Preprocessing Pipeline for IEEE-CIS Fraud Detection.
    
    Principles:
    1. Fit on train split only: Encoders, category vocabularies, and reference statistics
       are learned strictly from training data.
    2. Zero lookahead leakage: Time-based velocity and recency features are calculated
       chronologically forward in time.
    3. Interpretable features: Emphasizes behavioral velocities, recency, and amount distributions
       over hundreds of black-box V-columns.
    4. Robust unseen handling: Test categories not present in train map to a dedicated 'UNSEEN' category code.
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        time_col: str = "TransactionDT",
        id_col: str = "TransactionID",
        target_col: str = "isFraud",
        amt_col: str = "TransactionAmt"
    ):
        self.categorical_cols = categorical_cols or DEFAULT_CATEGORICAL_COLS
        self.time_col = time_col
        self.id_col = id_col
        self.target_col = target_col
        self.amt_col = amt_col
        
        # Learned artifacts (fitted on train only)
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.card_amount_stats: Dict[str, float] = {}
        self.global_amt_median: float = 0.0
        self.fitted_feature_names: List[str] = []
        self.is_fitted: bool = False

    @staticmethod
    def _create_card_proxy(df: pd.DataFrame) -> pd.Series:
        """Constructs an interpretable card/account identifier proxy."""
        cols_present = [c for c in CARD_ID_COLS if c in df.columns]
        if not cols_present:
            return pd.Series("CARD_UNKNOWN", index=df.index)
        
        # Combine present card identifiers
        card_series = df[cols_present[0]].astype(str)
        for c in cols_present[1:]:
            card_series = card_series + "_" + df[c].astype(str)
        return card_series

    @staticmethod
    def _create_device_proxy(df: pd.DataFrame) -> pd.Series:
        """Constructs an interpretable device identifier proxy."""
        cols_present = [c for c in DEVICE_ID_COLS if c in df.columns]
        if not cols_present:
            return pd.Series("DEVICE_UNKNOWN", index=df.index)
        
        device_series = df[cols_present[0]].astype(str)
        for c in cols_present[1:]:
            device_series = device_series + "_" + df[c].astype(str)
        return device_series

    def _compute_chronological_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes forward-looking rolling velocities and time-since-last transaction
        in strict TransactionDT order without backward leakage.
        """
        df = df.copy()
        
        # Ensure chronological ordering
        df = df.sort_values(by=self.time_col, ascending=True).reset_index(drop=True)
        
        card_proxy = self._create_card_proxy(df)
        device_proxy = self._create_device_proxy(df)
        df["_card_proxy"] = card_proxy
        df["_device_proxy"] = device_proxy

        time_s = df[self.time_col].values
        amt_s = df[self.amt_col].values if self.amt_col in df.columns else np.zeros(len(df))

        # Time-since-last transaction per card & device (in seconds)
        time_since_last_card = np.full(len(df), np.nan, dtype=np.float32)
        time_since_last_device = np.full(len(df), np.nan, dtype=np.float32)
        
        # Velocity counters (last 10 mins = 600s, last 1 hr = 3600s, last 24 hrs = 86400s)
        card_count_10m = np.zeros(len(df), dtype=np.int32)
        card_count_1h = np.zeros(len(df), dtype=np.int32)
        card_count_24h = np.zeros(len(df), dtype=np.int32)
        card_amt_sum_24h = np.zeros(len(df), dtype=np.float32)

        last_card_time: Dict[str, float] = {}
        last_device_time: Dict[str, float] = {}
        card_history: Dict[str, List[Tuple[float, float]]] = {}

        for i in range(len(df)):
            t = time_s[i]
            amt = amt_s[i]
            card = card_proxy.iloc[i]
            dev = device_proxy.iloc[i]

            # 1. Recency
            if card in last_card_time:
                time_since_last_card[i] = t - last_card_time[card]
            last_card_time[card] = t

            if dev != "DEVICE_UNKNOWN" and dev in last_device_time:
                time_since_last_device[i] = t - last_device_time[dev]
            if dev != "DEVICE_UNKNOWN":
                last_device_time[dev] = t

            # 2. Velocity windows for card
            if card not in card_history:
                card_history[card] = []
            
            # Prune events older than 24 hours (86400s)
            cutoff_24h = t - 86400.0
            history = [entry for entry in card_history[card] if entry[0] >= cutoff_24h]
            
            # Compute rolling counts within windows (excluding current transaction)
            c_10m = 0
            c_1h = 0
            c_24h = len(history)
            amt_24h = sum(entry[1] for entry in history)

            cutoff_10m = t - 600.0
            cutoff_1h = t - 3600.0

            for entry_t, _ in history:
                if entry_t >= cutoff_10m:
                    c_10m += 1
                if entry_t >= cutoff_1h:
                    c_1h += 1

            card_count_10m[i] = c_10m
            card_count_1h[i] = c_1h
            card_count_24h[i] = c_24h
            card_amt_sum_24h[i] = amt_24h

            # Append current transaction to history
            history.append((t, amt))
            card_history[card] = history

        # Add velocity columns
        df["time_since_last_txn_card"] = time_since_last_card
        df["time_since_last_txn_device"] = time_since_last_device
        df["card_txn_count_10m"] = card_count_10m
        df["card_txn_count_1h"] = card_count_1h
        df["card_txn_count_24h"] = card_count_24h
        df["card_amt_sum_24h"] = card_amt_sum_24h

        # Fill missing recency with -1 (meaning first transaction seen)
        df["time_since_last_txn_card"] = df["time_since_last_txn_card"].fillna(-1)
        df["time_since_last_txn_device"] = df["time_since_last_txn_device"].fillna(-1)

        # Cleanup temporary proxies
        df = df.drop(columns=["_card_proxy", "_device_proxy"])
        return df

    def fit(self, train_df: pd.DataFrame) -> "RiskFeaturePipeline":
        """
        Fits category vocabularies and reference statistics strictly on the train split.
        """
        logger.info("Fitting RiskFeaturePipeline on training data only...")
        
        # 1. Learn categorical encodings (train vocabulary only)
        self.category_mappings = {}
        for col in self.categorical_cols:
            if col in train_df.columns:
                # Treat NaN as explicit 'MISSING' category
                train_vals = train_df[col].fillna("MISSING").astype(str).unique()
                mapping = {"MISSING": 0, "UNSEEN": 1}
                curr_id = 2
                for val in train_vals:
                    if val not in mapping:
                        mapping[val] = curr_id
                        curr_id += 1
                self.category_mappings[col] = mapping
                logger.debug(f"Learned {len(mapping)} categories for '{col}' from train.")

        # 2. Learn reference statistics for transaction amounts
        if self.amt_col in train_df.columns:
            self.global_amt_median = float(train_df[self.amt_col].median())
            
            # Card mean amounts on train only
            card_proxy = self._create_card_proxy(train_df)
            train_with_card = train_df[[self.amt_col]].copy()
            train_with_card["_card_proxy"] = card_proxy
            self.card_amount_stats = train_with_card.groupby("_card_proxy")[self.amt_col].mean().to_dict()

        self.is_fitted = True
        logger.info(f"✔ Pipeline successfully fitted on {len(train_df):,} training records.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms dataset applying train-fitted encodings, amount ratios, and forward-looking velocities.
        """
        if not self.is_fitted:
            raise RuntimeError("RiskFeaturePipeline must be fitted on train data before transform.")

        logger.info(f"Transforming dataset with {len(df):,} records...")
        
        # 1. Compute chronological velocities
        transformed = self._compute_chronological_velocities(df)

        # 2. Transform categorical features using train mappings (unseen -> 'UNSEEN' = 1)
        for col, mapping in self.category_mappings.items():
            if col in transformed.columns:
                col_str = transformed[col].fillna("MISSING").astype(str)
                # Map values; fallback to 1 (UNSEEN)
                transformed[col + "_encoded"] = col_str.map(lambda v: mapping.get(v, 1)).astype(np.int32)

        # 3. Transaction Amount & Distribution features
        if self.amt_col in transformed.columns:
            amt = transformed[self.amt_col].fillna(self.global_amt_median)
            transformed["TransactionAmt_log"] = np.log1p(amt).astype(np.float32)
            transformed["TransactionAmt_decimal"] = (amt - np.floor(amt)).astype(np.float32)
            
            # Card mean ratio using train-learned card statistics
            card_proxy = self._create_card_proxy(transformed)
            card_means = card_proxy.map(self.card_amount_stats).fillna(self.global_amt_median).values
            # Avoid division by zero
            card_means = np.where(card_means <= 0, self.global_amt_median, card_means)
            transformed["amt_to_card_mean_ratio"] = (amt.values / card_means).astype(np.float32)

        # 4. Domain matching feature (if P_emaildomain and R_emaildomain exist)
        if "P_emaildomain" in transformed.columns and "R_emaildomain" in transformed.columns:
            p_email = transformed["P_emaildomain"].fillna("MISSING").astype(str)
            r_email = transformed["R_emaildomain"].fillna("MISSING").astype(str)
            transformed["is_same_email_domain"] = (
                (p_email == r_email) & (p_email != "MISSING")
            ).astype(np.int8)

        # 5. High-risk email flag (e.g. protonmail/anonymous domains)
        if "P_emaildomain" in transformed.columns:
            high_risk_domains = {"protonmail.com", "mail.com", "anonymous.com"}
            transformed["is_high_risk_email"] = (
                transformed["P_emaildomain"].astype(str).str.lower().isin(high_risk_domains)
            ).astype(np.int8)

        return transformed

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns list of engineered and model-ready feature column names."""
        exclude_cols = {
            self.id_col,
            self.time_col,
            self.target_col,
            "_card_proxy",
            "_device_proxy"
        }
        # Exclude raw string categorical columns if encoded version exists
        for col in self.categorical_cols:
            if col + "_encoded" in df.columns:
                exclude_cols.add(col)
                
        # Also exclude raw object/string columns
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c].dtype)
        ]
        return feature_cols


def run_layer2_pipeline(
    train_parquet_path: str = "data/processed/train.parquet",
    test_parquet_path: str = "data/processed/test.parquet",
    output_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Orchestrates Layer 2:
    1. Loads train.parquet and test.parquet from Layer 1.
    2. Fits RiskFeaturePipeline strictly on train split.
    3. Transforms train and test splits.
    4. Saves train_features.parquet, test_features.parquet, and feature pipeline artifacts.
    """
    if not os.path.exists(train_parquet_path) or not os.path.exists(test_parquet_path):
        raise FileNotFoundError(
            f"Layer 1 outputs not found at {train_parquet_path} or {test_parquet_path}. "
            "Please run Layer 1 first."
        )

    logger.info("=== Starting Layer 2: Feature Engineering & Preprocessing Pipeline ===")
    
    logger.info(f"Loading {train_parquet_path}...")
    train_df = pd.read_parquet(train_parquet_path)
    logger.info(f"Loading {test_parquet_path}...")
    test_df = pd.read_parquet(test_parquet_path)

    # Initialize and fit pipeline on train only
    pipeline = RiskFeaturePipeline()
    pipeline.fit(train_df)

    # Transform both splits
    train_features = pipeline.transform(train_df)
    test_features = pipeline.transform(test_df)

    feature_cols = pipeline.get_feature_columns(train_features)
    logger.info(f"Engineered {len(feature_cols)} model-ready features.")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    train_feat_path = os.path.join(output_dir, "train_features.parquet")
    test_feat_path = os.path.join(output_dir, "test_features.parquet")
    pipeline_path = os.path.join(output_dir, "feature_pipeline.joblib")
    meta_path = os.path.join(output_dir, "feature_metadata.json")

    logger.info(f"Saving transformed features to {train_feat_path} and {test_feat_path}...")
    train_features.to_parquet(train_feat_path, index=False, compression="snappy")
    test_features.to_parquet(test_feat_path, index=False, compression="snappy")

    logger.info(f"Saving fitted feature pipeline to {pipeline_path}...")
    joblib.dump(pipeline, pipeline_path)

    metadata = {
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "categorical_columns_encoded": list(pipeline.category_mappings.keys()),
        "global_amt_median": pipeline.global_amt_median,
        "artifact_paths": {
            "train_features": train_feat_path,
            "test_features": test_feat_path,
            "pipeline": pipeline_path
        }
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=== Layer 2 Complete: Output saved to %s ===", output_dir)
    return metadata
