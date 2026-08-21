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
import lightgbm as lgb

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

    feature_cols = [
        col for col in train_df.columns
        if col not in default_excludes and pd.api.types.is_numeric_dtype(train_df[col].dtype)
    ]

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
    Trains a Logistic Regression baseline with median imputation and standard scaling.
    Evaluates on test split to establish the true precision/recall floor.
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


def train_xgboost_classifier(
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
    Trains an XGBoost gradient-boosted decision tree classifier with scale_pos_weight.
    """
    logger.info("--- Training Step 2: XGBoost Gradient-Boosted Classifier ---")

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    if scale_pos_weight is None:
        scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

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

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_model_performance(y_test, y_pred, y_prob, model_name="XGBoost Classifier")
    metrics["scale_pos_weight_used"] = scale_pos_weight
    return clf, metrics


def train_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Trains a LightGBM gradient-boosted classifier with scale_pos_weight / is_unbalance.
    """
    logger.info("--- Training Step 3: LightGBM Gradient-Boosted Classifier ---")

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    if scale_pos_weight is None:
        scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=-1
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_model_performance(y_test, y_pred, y_prob, model_name="LightGBM Classifier")
    metrics["scale_pos_weight_used"] = scale_pos_weight
    return clf, metrics


def extract_gain_feature_importances(
    model: Any,
    feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Extracts gain-based feature importances from trained XGBoost or LightGBM model.
    """
    if isinstance(model, xgb.XGBClassifier):
        booster = model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        importances = []
        for i, col in enumerate(feature_names):
            gain_val = score_gain.get(col, score_gain.get(f"f{i}", 0.0))
            importances.append({"feature": col, "gain_importance": float(gain_val)})
    elif isinstance(model, lgb.LGBMClassifier):
        raw_importances = model.booster_.feature_importance(importance_type="gain")
        importances = [
            {"feature": col, "gain_importance": float(gain_val)}
            for col, gain_val in zip(feature_names, raw_importances)
        ]
    else:
        importances = [{"feature": col, "gain_importance": 0.0} for col in feature_names]

    importances.sort(key=lambda x: x["gain_importance"], reverse=True)
    return importances


def compare_models_honestly(
    baseline_metrics: Dict[str, Any],
    xgb_metrics: Dict[str, Any],
    lgb_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Directly compares GBDT models against baseline metrics.
    """
    comparison = {
        "baseline_pr_auc": baseline_metrics["pr_auc"],
        "xgb_pr_auc": xgb_metrics["pr_auc"],
        "xgb_pr_auc_delta": xgb_metrics["pr_auc"] - baseline_metrics["pr_auc"],
        "baseline_f1": baseline_metrics["f1_score"],
        "xgb_f1": xgb_metrics["f1_score"],
        "xgb_f1_delta": xgb_metrics["f1_score"] - baseline_metrics["f1_score"],
        "baseline_roc_auc": baseline_metrics["roc_auc"],
        "xgb_roc_auc": xgb_metrics["roc_auc"],
        "xgb_roc_auc_delta": xgb_metrics["roc_auc"] - baseline_metrics["roc_auc"]
    }

    if lgb_metrics:
        comparison.update({
            "lgb_pr_auc": lgb_metrics["pr_auc"],
            "lgb_pr_auc_delta": lgb_metrics["pr_auc"] - baseline_metrics["pr_auc"],
            "lgb_f1": lgb_metrics["f1_score"],
            "lgb_f1_delta": lgb_metrics["f1_score"] - baseline_metrics["f1_score"],
            "lgb_roc_auc": lgb_metrics["roc_auc"],
            "lgb_roc_auc_delta": lgb_metrics["roc_auc"] - baseline_metrics["roc_auc"]
        })

    summary = (
        f"PR-AUC -> Baseline: {baseline_metrics['pr_auc']:.4f} | "
        f"XGBoost: {xgb_metrics['pr_auc']:.4f} ({xgb_metrics['pr_auc'] - baseline_metrics['pr_auc']:+.4f})"
    )
    if lgb_metrics:
        summary += f" | LightGBM: {lgb_metrics['pr_auc']:.4f} ({lgb_metrics['pr_auc'] - baseline_metrics['pr_auc']:+.4f})"

    comparison["summary"] = summary
    logger.info(f"Model Comparison: {summary}")
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
    2. Trains Baseline (Logistic Regression).
    3. Trains XGBoost (with scale_pos_weight).
    4. Trains LightGBM (with scale_pos_weight).
    5. Evaluates and compares all three models.
    6. Extracts gain feature importances.
    7. Persists models and metrics JSON.
    """
    if not os.path.exists(train_features_path) or not os.path.exists(test_features_path):
        raise FileNotFoundError(
            f"Layer 2 feature files not found at {train_features_path} or {test_features_path}. "
            "Please run Layer 2 first."
        )

    logger.info("=== Starting Layer 3: Classifier Training & Baseline Comparison Pipeline ===")
    
    train_df = pd.read_parquet(train_features_path)
    test_df = pd.read_parquet(test_features_path)

    # 1. Prepare X and y
    X_train, y_train, X_test, y_test, feature_cols = prepare_feature_matrices(train_df, test_df)

    # 2. Train Baseline
    baseline_model, baseline_metrics = train_baseline_model(X_train, y_train, X_test, y_test)

    # 3. Train XGBoost
    xgb_model, xgb_metrics = train_xgboost_classifier(
        X_train, y_train, X_test, y_test,
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
    )

    # 4. Train LightGBM
    lgb_model, lgb_metrics = train_lightgbm_classifier(
        X_train, y_train, X_test, y_test,
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
    )

    # 5. Compare models
    comparison = compare_models_honestly(baseline_metrics, xgb_metrics, lgb_metrics)

    # 6. Extract Feature Importances (XGBoost)
    feature_importances = extract_gain_feature_importances(xgb_model, feature_cols)
    top_10_features = feature_importances[:10]

    # 7. Save Model Artifacts
    os.makedirs(output_dir, exist_ok=True)
    xgb_path = os.path.join(output_dir, "fraud_detector_gbdt.joblib")
    lgb_path = os.path.join(output_dir, "fraud_detector_lgbm.joblib")
    baseline_path = os.path.join(output_dir, "baseline_model.joblib")
    metrics_path = os.path.join(output_dir, "model_metrics.json")

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgb_model, lgb_path)
    joblib.dump(baseline_model, baseline_path)

    output_metadata = {
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "baseline_metrics": baseline_metrics,
        "xgboost_metrics": xgb_metrics,
        "lightgbm_metrics": lgb_metrics,
        "comparison": comparison,
        "top_features_by_gain": top_10_features,
        "all_feature_importances": feature_importances,
        "artifact_paths": {
            "xgboost_model": xgb_path,
            "lightgbm_model": lgb_path,
            "baseline_model": baseline_path,
            "metrics": metrics_path
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(output_metadata, f, indent=2)

    logger.info("=== Layer 3 Complete: Models and metrics saved to %s ===", output_dir)
    return output_metadata
