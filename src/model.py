import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "isFraud",
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Separates feature matrices (X) from targets (y) and drops identifier/non-feature columns.
    Ensures X contains strictly model-ready numeric and encoded columns.
    """
    default_excludes = {
        "TransactionID",
        "TransactionDT",
        target_col,
        "_card_proxy",
        "_device_proxy"
    }
    if exclude_cols:
        default_excludes.update(exclude_cols)

    # Candidate feature columns: numeric dtypes excluding identifiers
    feature_cols = [
        col for col in train_df.columns
        if col not in default_excludes and pd.api.types.is_numeric_dtype(train_df[col].dtype)
    ]

    # Ensure test_df has identical feature columns
    for c in feature_cols:
        if c not in test_df.columns:
            raise KeyError(f"Feature column '{c}' missing from test set.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int).copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(int).copy()

    logger.info(
        f"Feature matrices prepared: X_train shape {X_train.shape}, X_test shape {X_test.shape} with {len(feature_cols)} features."
    )
    return X_train, y_train, X_test, y_test, feature_cols


def evaluate_model_performance(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes precision, recall, F1, PR-AUC, and ROC-AUC on binary classification output.
    """
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Check if more than one class is present in true labels for AUC
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model_name": model_name,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "flags_per_true_fraud": float(fp / tp) if tp > 0 else float("inf")
    }

    logger.info(
        f"[{model_name}] Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}"
    )
    return metrics


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains a simple Logistic Regression baseline with median imputation and standard scaling.
    Evaluates on test split to establish the true performance floor before training GBDT.
    """
    logger.info("--- Training Step 1: Baseline Logistic Regression ---")
    
    baseline_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
    ])

    baseline_pipe.fit(X_train, y_train)

    y_pred = baseline_pipe.predict(X_test)
    y_prob = baseline_pipe.predict_proba(X_test)[:, 1]

    metrics = evaluate_model_performance(y_test, y_pred, y_prob, model_name="Baseline (Logistic Regression)")
    return baseline_pipe, metrics


def train_gbdt_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Trains an XGBoost gradient-boosted decision tree classifier with scale_pos_weight
    to handle the ~3.5% positive fraud class imbalance.
    """
    logger.info("--- Training Step 2: XGBoost Gradient-Boosted Classifier ---")

    # Compute scale_pos_weight if not provided: (negative_count / positive_count)
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    if scale_pos_weight is None:
        scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    logger.info(f"Class distribution: {neg_count:,} legit vs {pos_count:,} fraud.")
    logger.info(f"Using scale_pos_weight = {scale_pos_weight:.2f}")

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="aucpr",
        tree_method="hist"
    )

    clf.fit(X_train, y_train)

    # Score on held-out test split
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_model_performance(y_test, y_pred, y_prob, model_name="XGBoost Classifier")
    metrics["scale_pos_weight_used"] = scale_pos_weight
    return clf, metrics


def extract_gain_feature_importances(
    model: xgb.XGBClassifier,
    feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Extracts gain-based feature importances from trained XGBoost model.
    Gain measures the relative contribution of each feature to the model's tree splits.
    """
    booster = model.get_booster()
    # Get importance dictionary by gain
    score_gain = booster.get_score(importance_type="gain")
    
    # Map feature names (booster may use f0, f1 or actual names depending on input)
    importances = []
    for i, col in enumerate(feature_names):
        # Check both column name and 'f{i}' index
        gain_val = score_gain.get(col, score_gain.get(f"f{i}", 0.0))
        importances.append({
            "feature": col,
            "gain_importance": float(gain_val)
        })

    # Sort descending by gain
    importances.sort(key=lambda x: x["gain_importance"], reverse=True)
    return importances


def compare_models_honestly(
    baseline_metrics: Dict[str, Any],
    gbdt_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Directly compares GBDT against baseline metrics and documents relative deltas.
    """
    comparison = {
        "baseline_pr_auc": baseline_metrics["pr_auc"],
        "gbdt_pr_auc": gbdt_metrics["pr_auc"],
        "pr_auc_delta": gbdt_metrics["pr_auc"] - baseline_metrics["pr_auc"],
        "baseline_f1": baseline_metrics["f1_score"],
        "gbdt_f1": gbdt_metrics["f1_score"],
        "f1_delta": gbdt_metrics["f1_score"] - baseline_metrics["f1_score"],
        "baseline_roc_auc": baseline_metrics["roc_auc"],
        "gbdt_roc_auc": gbdt_metrics["roc_auc"],
        "roc_auc_delta": gbdt_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        "summary": (
            f"XGBoost PR-AUC: {gbdt_metrics['pr_auc']:.4f} vs Baseline: {baseline_metrics['pr_auc']:.4f} "
            f"(Delta: {gbdt_metrics['pr_auc'] - baseline_metrics['pr_auc']:+.4f})"
        )
    }
    logger.info(f"Model Comparison: {comparison['summary']}")
    return comparison


def run_layer3_pipeline(
    train_features_path: str = "data/processed/train_features.parquet",
    test_features_path: str = "data/processed/test_features.parquet",
    output_dir: str = "data/processed",
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Orchestrates Layer 3:
    1. Loads Layer 2 feature matrices.
    2. Separates X and y, dropping IDs and proxy markers.
    3. Trains Baseline model and evaluates precision/recall/PR-AUC.
    4. Trains XGBoost with scale_pos_weight.
    5. Evaluates and compares GBDT vs Baseline.
    6. Extracts gain-based feature importances.
    7. Persists trained models and metrics JSON.
    """
    if not os.path.exists(train_features_path) or not os.path.exists(test_features_path):
        raise FileNotFoundError(
            f"Layer 2 feature files not found at {train_features_path} or {test_features_path}. "
            "Please run Layer 2 first."
        )

    logger.info("=== Starting Layer 3: Classifier Training & Baseline Comparison Pipeline ===")
    
    logger.info(f"Loading {train_features_path}...")
    train_df = pd.read_parquet(train_features_path)
    logger.info(f"Loading {test_features_path}...")
    test_df = pd.read_parquet(test_features_path)

    # 1. Prepare X and y
    X_train, y_train, X_test, y_test, feature_cols = prepare_feature_matrices(train_df, test_df)

    # 2. Train Baseline
    baseline_model, baseline_metrics = train_baseline_model(X_train, y_train, X_test, y_test)

    # 3. Train GBDT Classifier
    gbdt_model, gbdt_metrics = train_gbdt_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    # 4. Compare models
    comparison = compare_models_honestly(baseline_metrics, gbdt_metrics)

    # 5. Extract Feature Importances
    feature_importances = extract_gain_feature_importances(gbdt_model, feature_cols)
    top_10_features = feature_importances[:10]
    logger.info(f"Top 10 Gain Features: {[f['feature'] for f in top_10_features]}")

    # 6. Save Model Artifacts
    os.makedirs(output_dir, exist_ok=True)
    gbdt_path = os.path.join(output_dir, "fraud_detector_gbdt.joblib")
    baseline_path = os.path.join(output_dir, "baseline_model.joblib")
    metrics_path = os.path.join(output_dir, "model_metrics.json")

    logger.info(f"Saving GBDT model to {gbdt_path}...")
    joblib.dump(gbdt_model, gbdt_path)

    logger.info(f"Saving Baseline model to {baseline_path}...")
    joblib.dump(baseline_model, baseline_path)

    output_metadata = {
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "baseline_metrics": baseline_metrics,
        "gbdt_metrics": gbdt_metrics,
        "comparison": comparison,
        "top_features_by_gain": top_10_features,
        "all_feature_importances": feature_importances,
        "artifact_paths": {
            "gbdt_model": gbdt_path,
            "baseline_model": baseline_path,
            "metrics": metrics_path
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(output_metadata, f, indent=2)

    logger.info(f"Saved model metrics and comparison to {metrics_path}.")
    logger.info("=== Layer 3 Complete: Models and metrics saved to %s ===", output_dir)
    return output_metadata
