import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import (
    prepare_feature_matrices,
    evaluate_model_performance,
    train_baseline_model,
    train_xgboost_classifier,
    train_lightgbm_classifier,
    extract_gain_feature_importances,
    compare_models_honestly,
    run_layer3_pipeline
)


@pytest.fixture
def synthetic_feature_splits():
    """Generates synthetic transformed feature DataFrames for training and testing."""
    np.random.seed(42)
    n_train, n_test = 200, 60
    
    def make_df(n, start_id, start_dt):
        is_fraud = np.zeros(n, dtype=int)
        is_fraud[:max(1, int(n * 0.05))] = 1
        np.random.shuffle(is_fraud)

        high_risk_signal = is_fraud * np.random.uniform(5, 10, size=n) + (1 - is_fraud) * np.random.uniform(0, 2, size=n)

        return pd.DataFrame({
            "TransactionID": np.arange(start_id, start_id + n),
            "TransactionDT": np.arange(start_dt, start_dt + n * 10, 10),
            "isFraud": is_fraud,
            "TransactionAmt": np.random.uniform(10, 500, size=n).astype(np.float32),
            "TransactionAmt_log": np.random.uniform(2, 6, size=n).astype(np.float32),
            "TransactionAmt_decimal": np.random.uniform(0, 1, size=n).astype(np.float32),
            "amt_to_expanding_card_mean_ratio": high_risk_signal.astype(np.float32),
            "card_txn_count_10m": (is_fraud * np.random.randint(3, 10, size=n)).astype(np.int32),
            "card_txn_count_1h": np.random.randint(1, 15, size=n).astype(np.int32),
            "card_txn_count_24h": np.random.randint(1, 30, size=n).astype(np.int32),
            "card_amt_sum_24h": np.random.uniform(50, 2000, size=n).astype(np.float32),
            "time_since_last_txn_card": np.random.uniform(10, 5000, size=n).astype(np.float32),
            "time_since_last_txn_device": np.random.uniform(-1, 5000, size=n).astype(np.float32),
            "ProductCD_encoded": np.random.randint(0, 5, size=n).astype(np.int32),
            "card4_encoded": np.random.randint(0, 4, size=n).astype(np.int32),
            "card6_encoded": np.random.randint(0, 3, size=n).astype(np.int32),
            "is_same_email_domain": np.random.randint(0, 2, size=n).astype(np.int8),
            "is_high_risk_email": np.random.randint(0, 2, size=n).astype(np.int8),
            "_card_proxy": ["card_proxy_" + str(i % 10) for i in range(n)],
            "_device_proxy": ["device_proxy_" + str(i % 5) for i in range(n)]
        })

    train_df = make_df(n_train, start_id=1000, start_dt=1000)
    test_df = make_df(n_test, start_id=5000, start_dt=50000)

    return train_df, test_df


def test_prepare_feature_matrices(synthetic_feature_splits):
    train_df, test_df = synthetic_feature_splits
    X_train, y_train, X_test, y_test, feature_cols = prepare_feature_matrices(train_df, test_df)

    assert "TransactionID" not in feature_cols
    assert "TransactionDT" not in feature_cols
    assert "isFraud" not in feature_cols
    assert "_card_proxy" not in feature_cols
    assert "_device_proxy" not in feature_cols

    assert X_train.shape[1] == len(feature_cols)
    assert X_test.shape[1] == len(feature_cols)
    assert len(y_train) == len(train_df)
    assert len(y_test) == len(test_df)


def test_train_baseline_model(synthetic_feature_splits):
    train_df, test_df = synthetic_feature_splits
    X_train, y_train, X_test, y_test, _ = prepare_feature_matrices(train_df, test_df)

    baseline_pipe, metrics = train_baseline_model(X_train, y_train, X_test, y_test)

    assert hasattr(baseline_pipe, "predict")
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_train_xgboost_and_lightgbm(synthetic_feature_splits):
    train_df, test_df = synthetic_feature_splits
    X_train, y_train, X_test, y_test, feature_cols = prepare_feature_matrices(train_df, test_df)

    # XGBoost
    xgb_clf, xgb_metrics = train_xgboost_classifier(
        X_train, y_train, X_test, y_test,
        n_estimators=30, max_depth=3, learning_rate=0.1
    )
    assert hasattr(xgb_clf, "predict")
    assert xgb_metrics["scale_pos_weight_used"] > 1.0

    # LightGBM
    lgb_clf, lgb_metrics = train_lightgbm_classifier(
        X_train, y_train, X_test, y_test,
        n_estimators=30, max_depth=3, learning_rate=0.1
    )
    assert hasattr(lgb_clf, "predict")
    assert lgb_metrics["scale_pos_weight_used"] > 1.0

    # Feature Importances
    xgb_imp = extract_gain_feature_importances(xgb_clf, feature_cols)
    lgb_imp = extract_gain_feature_importances(lgb_clf, feature_cols)
    assert len(xgb_imp) == len(feature_cols)
    assert len(lgb_imp) == len(feature_cols)


def test_compare_models_honestly(synthetic_feature_splits):
    train_df, test_df = synthetic_feature_splits
    X_train, y_train, X_test, y_test, _ = prepare_feature_matrices(train_df, test_df)

    _, base_metrics = train_baseline_model(X_train, y_train, X_test, y_test)
    _, xgb_metrics = train_xgboost_classifier(X_train, y_train, X_test, y_test, n_estimators=20)
    _, lgb_metrics = train_lightgbm_classifier(X_train, y_train, X_test, y_test, n_estimators=20)

    comparison = compare_models_honestly(base_metrics, xgb_metrics, lgb_metrics)
    assert "xgb_pr_auc_delta" in comparison
    assert "lgb_pr_auc_delta" in comparison


def test_end_to_end_layer3_pipeline(synthetic_feature_splits):
    train_df, test_df = synthetic_feature_splits
    with tempfile.TemporaryDirectory() as tmpdir:
        train_p = os.path.join(tmpdir, "train_features.parquet")
        test_p = os.path.join(tmpdir, "test_features.parquet")
        out_dir = os.path.join(tmpdir, "processed")

        train_df.to_parquet(train_p, index=False)
        test_df.to_parquet(test_p, index=False)

        results = run_layer3_pipeline(
            train_features_path=train_p,
            test_features_path=test_p,
            output_dir=out_dir,
            n_estimators=25,
            max_depth=3
        )

        assert os.path.exists(results["artifact_paths"]["xgboost_model"])
        assert os.path.exists(results["artifact_paths"]["lightgbm_model"])
        assert os.path.exists(results["artifact_paths"]["baseline_model"])
        assert os.path.exists(results["artifact_paths"]["metrics"])
