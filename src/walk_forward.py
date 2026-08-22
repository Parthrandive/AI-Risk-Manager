"""
Walk-Forward Temporal Robustness Engine

Implements:
1. Equal-time window chronological partitioning (5 periods across the ~182-day timeline).
2. Leak-free diagnostic model trained strictly on Periods 1–2.
3. Evaluation of the frozen model at increasing temporal distance (Period 3, 4, 5).
4. Rolling incremental retraining comparison (Retrain on P1–3 -> test on P4; Retrain on P1–4 -> test on P5)
   to quantify the exact empirical value of a production retraining cadence.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score
)

from src.feature_engineering import RiskFeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def partition_into_periods(
    df: pd.DataFrame,
    n_periods: int = 5,
    time_col: str = "TransactionDT"
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Partitions dataframe into equal-time chronological periods across the full timeline.
    Returns the dataframe with a 'period' column and metadata per period.
    """
    df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
    dt_min = float(df[time_col].min())
    dt_max = float(df[time_col].max())
    dt_span = dt_max - dt_min
    window_size = dt_span / n_periods

    # Assign period 1 to n_periods
    period_idx = np.minimum(n_periods, np.floor((df[time_col] - dt_min) / window_size).astype(int) + 1)
    df["period"] = period_idx

    period_metadata = {}
    for p in range(1, n_periods + 1):
        sub = df[df["period"] == p]
        tx_cnt = len(sub)
        fr_cnt = int(sub["isFraud"].sum()) if "isFraud" in sub.columns else 0
        fr_rate = (fr_cnt / tx_cnt * 100.0) if tx_cnt > 0 else 0.0
        p_min = int(sub[time_col].min()) if tx_cnt > 0 else 0
        p_max = int(sub[time_col].max()) if tx_cnt > 0 else 0
        span_days = round((p_max - p_min) / 86400.0, 1)

        period_metadata[p] = {
            "period": p,
            "transaction_count": tx_cnt,
            "fraud_count": fr_cnt,
            "fraud_rate_pct": round(fr_rate, 3),
            "dt_min": p_min,
            "dt_max": p_max,
            "span_days": span_days
        }

    return df, period_metadata


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive ranking, calibration, and threshold metrics."""
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = float(auc(recall_curve, precision_curve))
    roc_auc_val = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    y_pred_05 = (y_prob >= 0.5).astype(int)
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))

    # Calibrated operating point: threshold where review queue <= 3.0%
    thresholds = np.arange(0.10, 0.90, 0.01)
    best_op = {"threshold": 0.20, "precision": 0.0, "recall": 0.0, "flagged_pct": 0.0}
    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        flag_cnt = int(np.sum(pred_t))
        flag_pct = flag_cnt / len(y_true) * 100.0
        if flag_pct <= 3.0:
            tp = int(np.sum((y_true == 1) & (pred_t == 1)))
            prec = tp / flag_cnt if flag_cnt > 0 else 0.0
            rec = tp / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
            best_op = {
                "threshold": round(float(t), 2),
                "precision": round(float(prec) * 100.0, 2),
                "recall": round(float(rec) * 100.0, 2),
                "flagged_pct": round(float(flag_pct), 2)
            }
            break

    return {
        "pr_auc": round(pr_auc_val, 4),
        "roc_auc": round(roc_auc_val, 4),
        "brier_score": round(brier, 4),
        "precision_at_05": round(prec_05 * 100.0, 2),
        "recall_at_05": round(rec_05 * 100.0, 2),
        "f1_at_05": round(f1_05, 4),
        "calibrated_3pct_policy": best_op
    }


def run_walk_forward_experiment(
    train_parquet_path: str = "data/processed/train.parquet",
    test_parquet_path: str = "data/processed/test.parquet",
    output_dir: str = "data/processed",
    n_periods: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete walk-forward robustness proof:
    1. Partitions data into 5 equal-time periods.
    2. Trains frozen diagnostic model on Period 1–2.
    3. Evaluates frozen model on Period 3, Period 4, Period 5.
    4. Evaluates rolling incremental retraining on Period 4 and Period 5.
    """
    logger.info("=== Starting Step 2: Walk-Forward Temporal Robustness Experiment ===")
    
    # Load and combine full dataset
    train_df = pd.read_parquet(train_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    full_df, period_meta = partition_into_periods(full_df, n_periods=n_periods)

    # 1. Leak-Free Feature Engineering for Periods 1 through 5
    logger.info("Running leak-free continuous streaming feature engineering across all 5 periods...")
    p12_train = full_df[full_df["period"].isin([1, 2])].copy()
    
    pipeline = RiskFeaturePipeline().fit(p12_train)
    # Stream through full dataframe in temporal order
    full_transformed = pipeline.transform(full_df)

    exclude_cols = {"TransactionID", "TransactionDT", "isFraud", "period", "_card_proxy", "_device_proxy"}
    feature_cols = [c for c in full_transformed.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(full_transformed[c].dtype)]

    # 2. Train Frozen Diagnostic Model on Period 1–2
    logger.info("Training Frozen Diagnostic Model on Period 1–2 (%d transactions, %d features)...", len(p12_train), len(feature_cols))
    X_p12 = full_transformed[full_transformed["period"].isin([1, 2])][feature_cols]
    y_p12 = full_transformed[full_transformed["period"].isin([1, 2])]["isFraud"].values

    frozen_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        random_state=random_state,
        n_jobs=-1
    )
    frozen_model.fit(X_p12, y_p12)

    # 3. Score Frozen Model on Period 3, 4, 5
    frozen_results = {}
    for p in [3, 4, 5]:
        sub_df = full_transformed[full_transformed["period"] == p]
        X_p = sub_df[feature_cols]
        y_p = sub_df["isFraud"].values
        y_prob = frozen_model.predict_proba(X_p)[:, 1]
        metrics = evaluate_predictions(y_p, y_prob)
        frozen_results[f"period_{p}"] = {
            "period": p,
            "temporal_distance_days": round((p - 2.5) * 36.4, 1),
            "metrics": metrics
        }
        logger.info("Frozen Model on Period %d -> PR-AUC: %.4f | ROC-AUC: %.4f | 3%%-Cap Recall: %.2f%%", p, metrics["pr_auc"], metrics["roc_auc"], metrics["calibrated_3pct_policy"]["recall"])

    # 4. Rolling Incremental Retraining (Bonus Proof)
    logger.info("Evaluating Rolling Incremental Retraining on Period 4 and Period 5...")
    rolling_results = {}

    # Retrain on P1-3, test on P4
    X_p13 = full_transformed[full_transformed["period"].isin([1, 2, 3])][feature_cols]
    y_p13 = full_transformed[full_transformed["period"].isin([1, 2, 3])]["isFraud"].values
    retrained_p13_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05, tree_method="hist",
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.0, random_state=random_state, n_jobs=-1
    ).fit(X_p13, y_p13)

    sub_p4 = full_transformed[full_transformed["period"] == 4]
    prob_p4_retrained = retrained_p13_model.predict_proba(sub_p4[feature_cols])[:, 1]
    metrics_p4_retrained = evaluate_predictions(sub_p4["isFraud"].values, prob_p4_retrained)
    rolling_results["period_4_retrained"] = {
        "training_periods": "1-3",
        "eval_period": 4,
        "metrics": metrics_p4_retrained,
        "pr_auc_lift_over_frozen": round(metrics_p4_retrained["pr_auc"] - frozen_results["period_4"]["metrics"]["pr_auc"], 4)
    }

    # Retrain on P1-4, test on P5
    X_p14 = full_transformed[full_transformed["period"].isin([1, 2, 3, 4])][feature_cols]
    y_p14 = full_transformed[full_transformed["period"].isin([1, 2, 3, 4])]["isFraud"].values
    retrained_p14_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05, tree_method="hist",
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.0, random_state=random_state, n_jobs=-1
    ).fit(X_p14, y_p14)

    sub_p5 = full_transformed[full_transformed["period"] == 5]
    prob_p5_retrained = retrained_p14_model.predict_proba(sub_p5[feature_cols])[:, 1]
    metrics_p5_retrained = evaluate_predictions(sub_p5["isFraud"].values, prob_p5_retrained)
    rolling_results["period_5_retrained"] = {
        "training_periods": "1-4",
        "eval_period": 5,
        "metrics": metrics_p5_retrained,
        "pr_auc_lift_over_frozen": round(metrics_p5_retrained["pr_auc"] - frozen_results["period_5"]["metrics"]["pr_auc"], 4)
    }

    # Export artifacts
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "walk_forward_summary.json")

    results_payload = {
        "total_dataset_transactions": len(full_df),
        "total_fraud_cases": int(full_df["isFraud"].sum()),
        "timeline_span_days": round(float(full_df["TransactionDT"].max() - full_df["TransactionDT"].min()) / 86400.0, 1),
        "period_partition_metadata": period_meta,
        "frozen_model_evaluation": frozen_results,
        "rolling_retrain_evaluation": rolling_results,
        "drift_summary": {
            "p3_pr_auc": frozen_results["period_3"]["metrics"]["pr_auc"],
            "p4_pr_auc": frozen_results["period_4"]["metrics"]["pr_auc"],
            "p5_pr_auc": frozen_results["period_5"]["metrics"]["pr_auc"],
            "total_frozen_pr_auc_decay": round(frozen_results["period_5"]["metrics"]["pr_auc"] - frozen_results["period_3"]["metrics"]["pr_auc"], 4),
            "p4_retrain_lift": rolling_results["period_4_retrained"]["pr_auc_lift_over_frozen"],
            "p5_retrain_lift": rolling_results["period_5_retrained"]["pr_auc_lift_over_frozen"]
        }
    }

    with open(summary_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("=== Walk-Forward Experiment Complete: Summary saved to %s ===", summary_path)
    return results_payload
