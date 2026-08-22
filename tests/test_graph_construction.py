"""
Tests for Temporal Relational Graph Construction and Embedding Generation.

Verifies:
1. Zero lookahead leakage: Every edge in the graph connects from an earlier transaction (timestamp <= t)
   to the current transaction at time t. No future transaction or edge exists in the graph.
2. Isolated nodes: First-time transactions with no historical shared entities have degree 0 and distance -1.
3. State continuity across split boundary: Graph state seamlessly carries from train to test without resets.
4. Edge timestamp invariant: For every edge (u, v) in the generated edge list, timestamp(u) <= timestamp(v).
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_embeddings import (
    StreamingGraphState,
    TemporalGraphPipeline,
    extract_temporal_edges
)


@pytest.fixture
def synthetic_graph_txns():
    """
    Creates a deterministic sequence of transactions across time:
    - Txn 1 (t=100): Card A, Device X, Email E1 (first seen -> degree 0, no edges)
    - Txn 2 (t=200): Card B, Device Y, Email E2 (first seen -> degree 0, no edges)
    - Txn 3 (t=300): Card A, Device Y, Email E1 (Card A was seen at t=100; Device Y was seen at t=200)
      -> Edges to Txn 1 (via Card A, Email E1) and Txn 2 (via Device Y)
    - Txn 4 (t=400): Card A, Device Z, Email E3 (Card A was seen at t=100 and t=300)
      -> Edges to Txn 1 and Txn 3
    """
    return pd.DataFrame({
        "TransactionID": [101, 102, 103, 104],
        "TransactionDT": [100, 200, 300, 400],
        "card1": [1111, 2222, 1111, 1111],
        "card2": [1, 1, 1, 1],
        "card3": [1, 1, 1, 1],
        "card4": ["visa", "mastercard", "visa", "visa"],
        "card5": [1, 1, 1, 1],
        "card6": ["credit", "debit", "credit", "credit"],
        "DeviceType": ["mobile", "desktop", "desktop", "mobile"],
        "DeviceInfo": ["iOS", "Windows", "Windows", "Android"],
        "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", "anonymous.com"],
        "R_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", "anonymous.com"]
    })


def test_edge_timestamp_invariant(synthetic_graph_txns):
    """Asserts that for EVERY edge in the graph, source_timestamp <= target_timestamp."""
    edges = extract_temporal_edges(synthetic_graph_txns)
    
    assert len(edges) > 0, "Graph should contain temporal edges for shared entities."

    ts_map = dict(zip(synthetic_graph_txns["TransactionID"], synthetic_graph_txns["TransactionDT"]))

    for src_id, dst_id, relation in edges:
        t_src = ts_map[src_id]
        t_dst = ts_map[dst_id]
        # Invariant: source must be strictly strictly prior or simultaneous to target
        assert t_src <= t_dst, f"Leakage detected! Edge from t={t_src} (ID {src_id}) to t={t_dst} (ID {dst_id})"
        assert src_id != dst_id, "Self-loops are not valid temporal edges."


def test_first_node_isolation(synthetic_graph_txns):
    """Asserts that the first transaction on an entity has degree 0 and distance -1 (no lookahead)."""
    pipeline = TemporalGraphPipeline()
    feats = pipeline.transform(synthetic_graph_txns)

    # Txn 1 (Row 0): Brand new card, device, email
    assert feats.loc[0, "graph_deg_card"] == 0
    assert feats.loc[0, "graph_deg_device"] == 0
    assert feats.loc[0, "graph_deg_email"] == 0
    assert feats.loc[0, "graph_time_since_last_neighbor"] == -1.0

    # Txn 2 (Row 1): Different new card, device, email
    assert feats.loc[1, "graph_deg_card"] == 0
    assert feats.loc[1, "graph_deg_device"] == 0
    assert feats.loc[1, "graph_deg_email"] == 0
    assert feats.loc[1, "graph_time_since_last_neighbor"] == -1.0

    # Txn 3 (Row 2): Card A was seen at t=100 (degree=1), Device Y at t=200 (degree=1)
    assert feats.loc[2, "graph_deg_card"] == 1
    assert feats.loc[2, "graph_deg_device"] == 1
    assert feats.loc[2, "graph_time_since_last_neighbor"] == 100.0 # 300 - 200 = 100s (most recent neighbor was Txn 2)

    # Txn 4 (Row 3): Card A was seen at t=100 and t=300 (degree=2)
    assert feats.loc[3, "graph_deg_card"] == 2
    assert feats.loc[3, "graph_deg_device"] == 0 # Android is new
    assert feats.loc[3, "graph_time_since_last_neighbor"] == 100.0 # 400 - 300 = 100s (most recent neighbor was Txn 3)


def test_state_continuity_across_splits():
    """Asserts that graph connectivity carries seamlessly across train/test splits."""
    train_df = pd.DataFrame({
        "TransactionID": [101],
        "TransactionDT": [1000],
        "card1": [9999],
        "card2": [1], "card3": [1], "card4": ["visa"], "card5": [1], "card6": ["credit"],
        "DeviceType": ["mobile"], "DeviceInfo": ["iOS"],
        "P_emaildomain": ["gmail.com"], "R_emaildomain": ["gmail.com"]
    })

    test_df = pd.DataFrame({
        "TransactionID": [201],
        "TransactionDT": [1500],
        "card1": [9999],
        "card2": [1], "card3": [1], "card4": ["visa"], "card5": [1], "card6": ["credit"],
        "DeviceType": ["mobile"], "DeviceInfo": ["iOS"],
        "P_emaildomain": ["gmail.com"], "R_emaildomain": ["gmail.com"]
    })

    pipeline = TemporalGraphPipeline()
    train_feats, test_feats = pipeline.transform_splits(train_df, test_df)

    # Train row 0: isolated
    assert train_feats.loc[0, "graph_deg_card"] == 0
    assert train_feats.loc[0, "graph_time_since_last_neighbor"] == -1.0

    # Test row 0: connects to Train row 0 (dt = 500s)
    assert test_feats.loc[0, "graph_deg_card"] == 1
    assert test_feats.loc[0, "graph_deg_device"] == 1
    assert test_feats.loc[0, "graph_time_since_last_neighbor"] == 500.0
