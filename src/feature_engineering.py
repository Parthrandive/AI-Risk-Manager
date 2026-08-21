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

# Canonical IEEE-CIS card and device proxy components
# Card Proxy: Combines card bin, issue parameters, and billing address
CARD_ID_COLS = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"]
# Device Proxy: Combines hardware, OS, and browser identity fields
DEVICE_ID_COLS = ["DeviceType", "DeviceInfo", "id_30", "id_31"]


class StreamingRiskState:
    """
    Maintains time-ordered streaming state for rolling velocities, recency,
    and expanding card statistics across train and test temporal boundaries.
    """

    def __init__(self):
        # Maps card_proxy -> list of (timestamp, amount) within sliding 24h window
        self.card_history: Dict[str, List[Tuple[float, float]]] = {}
        # Maps card_proxy -> last seen timestamp
        self.last_card_time: Dict[str, float] = {}
        # Maps device_proxy -> last seen timestamp
        self.last_device_time: Dict[str, float] = {}
        # Maps card_proxy -> (cumulative_amount, transaction_count) for expanding mean
        self.card_expanding_stats: Dict[str, Tuple[float, int]] = {}
        # Global expanding stats: (cumulative_amount, transaction_count)
        self.global_expanding_stats: Tuple[float, int] = (0.0, 0)


class RiskFeaturePipeline:
    """
    Leak-Free, Continuous-Stream Feature Engineering Pipeline for IEEE-CIS Fraud Detection.
    
    Key Design Principles:
    1. Fit on Train Only:
       - Categorical vocabularies and imputation references are learned strictly from train.parquet.
       - Novel categories in test map to a distinct 'UNSEEN' token.
    2. Continuous State Across Split Boundary (No Cold-Start Artifacts):
       - Rolling velocities (10m, 1h, 24h) and recency carry state across the split boundary.
       - Test transactions immediately following the 80th percentile split retain full 24h context.
    3. Expanding Historical Card Mean (Zero Intra-Split Lookahead Leakage):
       - Card amount ratios use cumulative expanding means up to timestamp t-1 (never global future means).
    4. Interpretable Risk Proxies:
       - Card Proxy: card1–card6 + addr1 + addr2
       - Device Proxy: DeviceType + DeviceInfo + id_30 + id_31
       (Note: DeviceInfo is ~70-80% null in IEEE-CIS; recency defaults to -1 when absent).
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
        
        # Learned vocabulary mappings (fitted on train only)
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.global_amt_median: float = 0.0
        self.is_fitted: bool = False
        
        # Continuous streaming state
        self.state: StreamingRiskState = StreamingRiskState()

    @staticmethod
    def create_card_proxy(df: pd.DataFrame) -> pd.Series:
        """Constructs canonical card/account proxy: card1-card6 + addr1-addr2."""
        cols_present = [c for c in CARD_ID_COLS if c in df.columns]
        if not cols_present:
            return pd.Series("CARD_UNKNOWN", index=df.index)
        
        card_series = df[cols_present[0]].fillna("NA").astype(str)
        for c in cols_present[1:]:
            card_series = card_series + "_" + df[c].fillna("NA").astype(str)
        return card_series

    @staticmethod
    def create_device_proxy(df: pd.DataFrame) -> pd.Series:
        """Constructs device proxy: DeviceType + DeviceInfo + id_30 + id_31."""
        cols_present = [c for c in DEVICE_ID_COLS if c in df.columns]
        if not cols_present:
            return pd.Series("DEVICE_UNKNOWN", index=df.index)
        
        device_series = df[cols_present[0]].fillna("NA").astype(str)
        for c in cols_present[1:]:
            device_series = device_series + "_" + df[c].fillna("NA").astype(str)
        # If all fields were NA, mark as DEVICE_UNKNOWN
        all_na_str = "_".join(["NA"] * len(cols_present))
        device_series = device_series.replace(all_na_str, "DEVICE_UNKNOWN")
        return device_series

    def fit(self, train_df: pd.DataFrame) -> "RiskFeaturePipeline":
        """
        Fits categorical vocabularies and global references strictly on the train split.
        """
        logger.info("Fitting RiskFeaturePipeline on training split vocabulary...")
        
        # 1. Learn categorical encodings (train vocabulary only)
        self.category_mappings = {}
        for col in self.categorical_cols:
            if col in train_df.columns:
                train_vals = train_df[col].fillna("MISSING").astype(str).unique()
                mapping = {"MISSING": 0, "UNSEEN": 1}
                curr_id = 2
                for val in train_vals:
                    if val not in mapping:
                        mapping[val] = curr_id
                        curr_id += 1
                self.category_mappings[col] = mapping
                logger.debug(f"Learned {len(mapping)} categories for '{col}' from train.")

        # 2. Learn global amount median on train
        if self.amt_col in train_df.columns:
            self.global_amt_median = float(train_df[self.amt_col].median())
        else:
            self.global_amt_median = 50.0

        # Reset streaming state
        self.state = StreamingRiskState()
        self.is_fitted = True
        logger.info(f"✔ Pipeline vocabulary fitted on {len(train_df):,} training records.")
        return self

    def _transform_chunk(
        self,
        df: pd.DataFrame,
        update_state: bool = True
    ) -> pd.DataFrame:
        """
        Transforms a chronological chunk of transactions using current streaming state.
        Computes rolling velocities, recencies, and expanding card statistics forward in time.
        """
        df = df.copy()
        
        # Sort ascending by TransactionDT
        df = df.sort_values(by=self.time_col, ascending=True).reset_index(drop=True)
        
        card_proxy = self.create_card_proxy(df)
        device_proxy = self.create_device_proxy(df)
        
        time_s = df[self.time_col].values
        amt_s = df[self.amt_col].fillna(self.global_amt_median).values if self.amt_col in df.columns else np.zeros(len(df))

        n = len(df)
        time_since_last_card = np.full(n, -1.0, dtype=np.float32)
        time_since_last_device = np.full(n, -1.0, dtype=np.float32)
        card_count_10m = np.zeros(n, dtype=np.int32)
        card_count_1h = np.zeros(n, dtype=np.int32)
        card_count_24h = np.zeros(n, dtype=np.int32)
        card_amt_sum_24h = np.zeros(n, dtype=np.float32)
        amt_to_card_expanding_ratio = np.ones(n, dtype=np.float32)

        st = self.state

        for i in range(n):
            t = float(time_s[i])
            amt = float(amt_s[i])
            card = card_proxy.iloc[i]
            dev = device_proxy.iloc[i]

            # 1. Recency
            if card in st.last_card_time:
                time_since_last_card[i] = t - st.last_card_time[card]
            
            if dev != "DEVICE_UNKNOWN" and dev in st.last_device_time:
                time_since_last_device[i] = t - st.last_device_time[dev]

            # 2. Expanding Card Mean (Strictly historical up to t-1, no future leakage)
            if card in st.card_expanding_stats:
                cum_amt, count = st.card_expanding_stats[card]
                historical_card_mean = cum_amt / count
            elif st.global_expanding_stats[1] > 0:
                historical_card_mean = st.global_expanding_stats[0] / st.global_expanding_stats[1]
            else:
                historical_card_mean = self.global_amt_median

            if historical_card_mean > 0:
                amt_to_card_expanding_ratio[i] = amt / historical_card_mean
            else:
                amt_to_card_expanding_ratio[i] = 1.0

            # 3. Rolling Velocity windows (last 10m=600s, 1h=3600s, 24h=86400s)
            if card not in st.card_history:
                st.card_history[card] = []
            
            cutoff_24h = t - 86400.0
            # Retain only events within the last 24 hours
            history = [entry for entry in st.card_history[card] if entry[0] >= cutoff_24h]
            
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

            # Update state for next transactions if requested
            if update_state:
                st.last_card_time[card] = t
                if dev != "DEVICE_UNKNOWN":
                    st.last_device_time[dev] = t

                history.append((t, amt))
                st.card_history[card] = history

                # Update expanding statistics
                prev_cum, prev_cnt = st.card_expanding_stats.get(card, (0.0, 0))
                st.card_expanding_stats[card] = (prev_cum + amt, prev_cnt + 1)
                
                g_cum, g_cnt = st.global_expanding_stats
                st.global_expanding_stats = (g_cum + amt, g_cnt + 1)

        # Attach engineered velocity and behavioral features
        df["time_since_last_txn_card"] = time_since_last_card
        df["time_since_last_txn_device"] = time_since_last_device
        df["card_txn_count_10m"] = card_count_10m
        df["card_txn_count_1h"] = card_count_1h
        df["card_txn_count_24h"] = card_count_24h
        df["card_amt_sum_24h"] = card_amt_sum_24h
        df["amt_to_expanding_card_mean_ratio"] = amt_to_card_expanding_ratio

        # 4. Amount representations
        if self.amt_col in df.columns:
            amt_series = df[self.amt_col].fillna(self.global_amt_median)
            df["TransactionAmt_log"] = np.log1p(amt_series).astype(np.float32)
            df["TransactionAmt_decimal"] = (amt_series - np.floor(amt_series)).astype(np.float32)

        # 5. Transform categorical features using train mappings (unseen -> 'UNSEEN' = 1)
        for col, mapping in self.category_mappings.items():
            if col in df.columns:
                col_str = df[col].fillna("MISSING").astype(str)
                df[col + "_encoded"] = col_str.map(lambda v: mapping.get(v, 1)).astype(np.int32)

        # 6. Domain matching & high-risk email checks
        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            p_email = df["P_emaildomain"].fillna("MISSING").astype(str)
            r_email = df["R_emaildomain"].fillna("MISSING").astype(str)
            df["is_same_email_domain"] = (
                (p_email == r_email) & (p_email != "MISSING")
            ).astype(np.int8)

        if "P_emaildomain" in df.columns:
            high_risk_domains = {"protonmail.com", "mail.com", "anonymous.com"}
            df["is_high_risk_email"] = (
                df["P_emaildomain"].astype(str).str.lower().isin(high_risk_domains)
            ).astype(np.int8)

        return df

    def transform_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms train and test splits in seamless chronological succession.
        Carries forward rolling state across the split boundary to prevent test cold-start artifacts.
        """
        if not self.is_fitted:
            raise RuntimeError("RiskFeaturePipeline must be fitted before calling transform_splits.")

        logger.info("Transforming train split and updating streaming state...")
        # Reset state to ensure fresh start from beginning of train
        self.state = StreamingRiskState()
        train_transformed = self._transform_chunk(train_df, update_state=True)

        logger.info(
            "Transforming test split with continuous state carryover across boundary..."
        )
        # Test picks up directly from the end-state of train
        test_transformed = self._transform_chunk(test_df, update_state=True)

        return train_transformed, test_transformed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms an independent dataframe/batch using current streaming state.
        """
        if not self.is_fitted:
            raise RuntimeError("RiskFeaturePipeline must be fitted before transform.")
        return self._transform_chunk(df, update_state=True)

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns list of model-ready numeric and encoded feature column names."""
        exclude_cols = {
            self.id_col,
            self.time_col,
            self.target_col
        }
        for col in self.categorical_cols:
            if col + "_encoded" in df.columns:
                exclude_cols.add(col)
                
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
    1. Ingests train.parquet and test.parquet from Layer 1.
    2. Fits vocabularies and global references strictly on train.parquet.
    3. Transforms train and test with continuous time-series state across boundary.
    4. Computes expanding card mean ratios and rolling velocity windows.
    5. Saves parquet features and fitted pipeline artifacts.
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

    # Transform both splits with seamless state progression across boundary
    train_features, test_features = pipeline.transform_splits(train_df, test_df)

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
        "proxy_definitions": {
            "card_proxy": "card1 + card2 + card3 + card4 + card5 + card6 + addr1 + addr2",
            "device_proxy": "DeviceType + DeviceInfo + id_30 + id_31 (DeviceInfo ~70-80% null, recency defaults to -1)"
        },
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
