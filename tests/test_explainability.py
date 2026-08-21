import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.explainability import (
    calibrate_gateway_thresholds,
    RiskExplainerGateway,
    run_layer5_pipeline
)


@pytest.fixture
def synthetic_triage_data():
    """Generates synthetic predictions and labels for gateway calibration."""
    np.random.seed(42)
    n = 1000
    y_true = np.zeros(n, dtype=int)
    y_true[:50] = 1 # 5% fraud

    # Correlated probabilities
    y_prob = np.where(y_true == 1, np.random.uniform(0.3, 0.95, size=n), np.random.uniform(0.01, 0.4, size=n))
    return y_true, y_prob


def test_calibrate_gateway_thresholds(synthetic_triage_data):
    y_true, y_prob = synthetic_triage_data
    tau_low, tau_high, meta = calibrate_gateway_thresholds(
        y_prob=y_prob,
        y_true=y_true,
        max_review_budget_pct=3.0,
        min_autoblock_precision=90.0
    )

    assert 0.0 < tau_low < tau_high < 1.0
    assert "review_capacity_cap_pct" in meta
    assert "max_review_cases_budget" in meta


def test_risk_explainer_routing_and_card():
    # Train a toy model
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "TransactionAmt": np.random.uniform(10, 500, size=n).astype(np.float32),
        "amt_to_expanding_card_mean_ratio": np.random.uniform(0.5, 4.0, size=n).astype(np.float32),
        "card_count_10m": np.random.randint(0, 5, size=n).astype(np.int32),
        "V257": np.random.uniform(0, 10, size=n).astype(np.float32)
    })
    y = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    
    model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    gateway = RiskExplainerGateway(
        model=model,
        feature_names=list(X.columns),
        tau_low=0.20,
        tau_high=0.80
    )

    assert gateway.route_decision(0.10) == "AUTO_APPROVE"
    assert gateway.route_decision(0.40) == "MANUAL_REVIEW"
    assert gateway.route_decision(0.85) == "AUTO_BLOCK"

    card = gateway.explain_transaction(
        X_row=X.iloc[0],
        risk_score=0.45,
        transaction_id=1001
    )

    assert card["transaction_id"] == 1001
    assert card["decision"] == "MANUAL_REVIEW"
    assert "top_interpretable_factors" in card
    assert "opaque_signal_disclosure" in card
    assert "undisclosed_v_feature_contribution_pct" in card["opaque_signal_disclosure"]


def test_run_layer5_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(42)
        n = 200
        y = np.zeros(n, dtype=int)
        y[:10] = 1
        test_df = pd.DataFrame({
            "TransactionID": np.arange(1000, 1000 + n),
            "TransactionDT": np.arange(100, 100 + n * 10, 10),
            "isFraud": y,
            "TransactionAmt": np.random.uniform(10, 200, size=n).astype(np.float32),
            "amt_to_expanding_card_mean_ratio": np.random.uniform(0.5, 3.0, size=n).astype(np.float32),
            "card_count_10m": np.random.randint(0, 3, size=n).astype(np.int32),
            "V257": np.random.uniform(0, 5, size=n).astype(np.float32)
        })

        test_p = os.path.join(tmpdir, "test_features.parquet")
        test_df.to_parquet(test_p, index=False)

        feat_cols = ["TransactionAmt", "amt_to_expanding_card_mean_ratio", "card_count_10m", "V257"]
        model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
        model.fit(test_df[feat_cols], y)
        model_p = os.path.join(tmpdir, "model.joblib")
        joblib.dump(model, model_p)

        results = run_layer5_pipeline(
            test_features_path=test_p,
            model_path=model_p,
            output_dir=tmpdir,
            sample_audit_count=2
        )

        assert os.path.exists(results["artifact_paths"]["triage_queue_parquet"])
        assert os.path.exists(results["artifact_paths"]["triage_summary_json"])
        assert "triage_distribution" in results
        assert len(results["sample_audit_cards"]) > 0
