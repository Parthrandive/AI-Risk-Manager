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

from src.evaluation import (
    sweep_decision_thresholds,
    sweep_scale_pos_weight_impact,
    compute_operational_cost_curve,
    get_what_didnt_work_registry,
    run_layer4_pipeline
)


@pytest.fixture
def synthetic_eval_data():
    """Generates synthetic predictions and labels for evaluation testing."""
    np.random.seed(42)
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_true[:10] = 1 # 5% fraud
    np.random.shuffle(y_true)

    # Correlated probabilities
    y_prob = np.where(y_true == 1, np.random.uniform(0.4, 0.95, size=n), np.random.uniform(0.01, 0.5, size=n))
    amounts = np.random.uniform(10, 500, size=n)

    return y_true, y_prob, amounts


def test_sweep_decision_thresholds(synthetic_eval_data):
    y_true, y_prob, amounts = synthetic_eval_data
    sweep_df = sweep_decision_thresholds(y_true, y_prob, amounts=amounts, thresholds=[0.1, 0.5, 0.8])

    assert len(sweep_df) == 3
    assert "precision" in sweep_df.columns
    assert "recall" in sweep_df.columns
    assert "flags_per_true_fraud" in sweep_df.columns
    assert "caught_fraud_amount" in sweep_df.columns
    assert "fraud_dollar_capture_rate" in sweep_df.columns

    # Recall at low threshold >= recall at high threshold
    assert sweep_df.loc[0, "recall"] >= sweep_df.loc[2, "recall"]


def test_operational_cost_curve(synthetic_eval_data):
    y_true, y_prob, amounts = synthetic_eval_data
    sweep_df = sweep_decision_thresholds(y_true, y_prob, amounts=amounts)
    cost_df = compute_operational_cost_curve(sweep_df, cost_per_manual_review=5.0, fraud_chargeback_multiplier=1.5)

    assert "manual_review_cost" in cost_df.columns
    assert "net_financial_benefit" in cost_df.columns

    # Exact arithmetic verification across all rows
    expected_net = (cost_df["fraud_loss_prevented"] - cost_df["manual_review_cost"] - cost_df["missed_fraud_loss"]).round(2)
    assert np.allclose(cost_df["net_financial_benefit"], expected_net, atol=0.01)


def test_what_didnt_work_registry():
    registry = get_what_didnt_work_registry()
    assert len(registry) >= 4
    for entry in registry:
        assert "category" in entry
        assert "hypothesis_or_approach" in entry
        assert "outcome" in entry
        assert "root_cause_analysis" in entry


def test_run_layer4_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(42)
        n = 100
        y = np.zeros(n, dtype=int)
        y[:5] = 1
        test_df = pd.DataFrame({
            "TransactionID": np.arange(1000, 1000 + n),
            "TransactionDT": np.arange(100, 100 + n * 10, 10),
            "isFraud": y,
            "TransactionAmt": np.random.uniform(10, 200, size=n).astype(np.float32),
            "feat_1": np.random.uniform(0, 1, size=n).astype(np.float32),
            "feat_2": np.random.uniform(0, 10, size=n).astype(np.float32)
        })

        test_p = os.path.join(tmpdir, "test_features.parquet")
        test_df.to_parquet(test_p, index=False)

        # Train a toy model
        X = test_df[["TransactionAmt", "feat_1", "feat_2"]]
        model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X, y)
        model_p = os.path.join(tmpdir, "model.joblib")
        joblib.dump(model, model_p)

        results = run_layer4_pipeline(
            test_features_path=test_p,
            model_path=model_p,
            output_dir=tmpdir
        )

        assert os.path.exists(results["artifact_paths"]["threshold_sweep_parquet"])
        assert os.path.exists(results["artifact_paths"]["threshold_sweep_csv"])
        assert os.path.exists(results["artifact_paths"]["evaluation_summary"])
        assert os.path.exists(results["artifact_paths"]["what_didnt_work"])
