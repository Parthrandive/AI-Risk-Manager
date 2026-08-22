"""
Temporal Relational Graph Construction & Inductive Embedding Pipeline

Implements:
1. Strict temporal graph construction: Every edge (u, v) connects a strictly prior transaction u
   (timestamp <= t) to the current transaction v at timestamp t.
2. Inductive streaming representation: Generates N=8 relational graph features representing
   1-hop entity degrees, 2-hop identity breadth, and multi-entity temporal proximity.
3. Zero future lookahead & state continuity across train/test chronological boundaries.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_temporal_edges(df: pd.DataFrame) -> List[Tuple[int, int, str]]:
    """
    Extracts all directed temporal edges (src_txn_id, dst_txn_id, relation) from dataframe.
    Structurally guarantees that src_timestamp <= dst_timestamp for every edge.
    """
    df_sorted = df.sort_values(by="TransactionDT", ascending=True).reset_index(drop=True)
    
    entity_history: Dict[str, Tuple[int, float]] = {}
    edges: List[Tuple[int, int, str]] = []

    for _, row in df_sorted.iterrows():
        txn_id = int(row["TransactionID"])
        t_curr = float(row["TransactionDT"])

        # Extract entities
        entities = _extract_row_entities(row)

        for rel_type, ent_val in entities:
            if ent_val in entity_history:
                prev_id, prev_ts = entity_history[ent_val]
                if prev_id != txn_id and prev_ts <= t_curr:
                    edges.append((prev_id, txn_id, rel_type))
            
            # Update history strictly at or after processing
            entity_history[ent_val] = (txn_id, t_curr)

    return edges


def _extract_row_entities(row: pd.Series) -> List[Tuple[str, str]]:
    """Extracts valid entity identifiers from a transaction row."""
    entities = []

    # 1. Pure card instrument
    pure_card_parts = [str(row[c]) if c in row and pd.notna(row[c]) else "NA" for c in ["card1", "card2", "card3", "card4", "card5", "card6"]]
    pure_card_str = "CARD_" + "_".join(pure_card_parts)
    if not all(p == "NA" for p in pure_card_parts):
        entities.append(("card", pure_card_str))

    # 2. Device proxy
    dev_type = str(row["DeviceType"]) if "DeviceType" in row and pd.notna(row["DeviceType"]) else ""
    dev_info = str(row["DeviceInfo"]) if "DeviceInfo" in row and pd.notna(row["DeviceInfo"]) else ""
    if dev_type or dev_info:
        dev_str = f"DEV_{dev_type}_{dev_info}"
        entities.append(("device", dev_str))

    # 3. Email domain pair
    p_email = str(row["P_emaildomain"]) if "P_emaildomain" in row and pd.notna(row["P_emaildomain"]) else ""
    r_email = str(row["R_emaildomain"]) if "R_emaildomain" in row and pd.notna(row["R_emaildomain"]) else ""
    if p_email or r_email:
        email_str = f"EMAIL_{p_email}_{r_email}"
        entities.append(("email", email_str))

    return entities


class StreamingGraphState:
    """Maintains inductive relational graph state forward in time."""

    def __init__(self):
        self.entity_counts: Dict[str, int] = {}
        self.entity_last_ts: Dict[str, float] = {}
        self.card_linked_devices: Dict[str, Set[str]] = {}
        self.card_linked_emails: Dict[str, Set[str]] = {}

    def reset(self):
        self.entity_counts.clear()
        self.entity_last_ts.clear()
        self.card_linked_devices.clear()
        self.card_linked_emails.clear()


class TemporalGraphPipeline:
    """
    Inductive Temporal Graph Feature Engineering Pipeline.
    Generates N=8 relational features strictly up to t-1.
    """

    def __init__(self):
        self.state = StreamingGraphState()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a dataframe in temporal order while continuously updating graph state."""
        df_sorted = df.sort_values(by="TransactionDT", ascending=True).reset_index(drop=True)
        n = len(df_sorted)

        # Preallocate feature arrays
        deg_card = np.zeros(n, dtype=np.int32)
        deg_device = np.zeros(n, dtype=np.int32)
        deg_email = np.zeros(n, dtype=np.int32)
        breadth_device = np.zeros(n, dtype=np.int32)
        breadth_email = np.zeros(n, dtype=np.int32)
        time_since_neighbor = np.full(n, -1.0, dtype=np.float32)
        total_degree = np.zeros(n, dtype=np.int32)
        multi_identity = np.zeros(n, dtype=np.int32)

        for i in range(n):
            row = df_sorted.iloc[i]
            t_curr = float(row["TransactionDT"])
            entities = _extract_row_entities(row)

            # 1. Query current graph state up to t-1
            recent_deltas = []
            card_ent = None
            dev_ent = None
            email_ent = None

            for rel_type, ent_val in entities:
                cnt = self.state.entity_counts.get(ent_val, 0)
                last_ts = self.state.entity_last_ts.get(ent_val, None)

                if rel_type == "card":
                    deg_card[i] = cnt
                    card_ent = ent_val
                    breadth_device[i] = len(self.state.card_linked_devices.get(ent_val, set()))
                    breadth_email[i] = len(self.state.card_linked_emails.get(ent_val, set()))
                elif rel_type == "device":
                    deg_device[i] = cnt
                    dev_ent = ent_val
                elif rel_type == "email":
                    deg_email[i] = cnt
                    email_ent = ent_val

                if last_ts is not None and t_curr >= last_ts:
                    recent_deltas.append(t_curr - last_ts)

            total_degree[i] = deg_card[i] + deg_device[i] + deg_email[i]
            if recent_deltas:
                time_since_neighbor[i] = min(recent_deltas)

            # Multi-identity flag: transaction links to >1 distinct previously seen entity types
            active_entity_types_seen = (deg_card[i] > 0) + (deg_device[i] > 0) + (deg_email[i] > 0)
            if active_entity_types_seen >= 2:
                multi_identity[i] = 1

            # 2. Update graph state strictly at or after evaluating row i
            for rel_type, ent_val in entities:
                self.state.entity_counts[ent_val] = self.state.entity_counts.get(ent_val, 0) + 1
                self.state.entity_last_ts[ent_val] = t_curr

            if card_ent and dev_ent:
                if card_ent not in self.state.card_linked_devices:
                    self.state.card_linked_devices[card_ent] = set()
                self.state.card_linked_devices[card_ent].add(dev_ent)

            if card_ent and email_ent:
                if card_ent not in self.state.card_linked_emails:
                    self.state.card_linked_emails[card_ent] = set()
                self.state.card_linked_emails[card_ent].add(email_ent)

        out_df = pd.DataFrame({
            "graph_deg_card": deg_card,
            "graph_deg_device": deg_device,
            "graph_deg_email": deg_email,
            "graph_2hop_device_breadth": breadth_device,
            "graph_2hop_email_breadth": breadth_email,
            "graph_time_since_last_neighbor": time_since_neighbor,
            "graph_total_relational_degree": total_degree,
            "graph_multi_identity_active": multi_identity
        })

        return out_df

    def transform_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transforms train split, preserves graph state across boundary, and transforms test split."""
        self.state.reset()
        train_feats = self.transform(train_df)
        test_feats = self.transform(test_df)
        return train_feats, test_feats
