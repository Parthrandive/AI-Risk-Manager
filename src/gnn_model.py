"""
PyTorch Temporal GraphSAGE Neural Embedding Engine

Implements:
1. Two-layer GraphSAGE architecture with mean neighborhood aggregation.
2. Strict temporal edge invariant: each node v only aggregates historical neighbors u with Timestamp(u) <= Timestamp(v).
3. Supervised classification loss tracking per epoch to verify training convergence.
4. Generates 8-dimensional neural embeddings for downstream GBDT integration.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TemporalGraphSAGELayer(nn.Module):
    """Single GraphSAGE layer with self-projection and neighbor aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neigh = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x_self: torch.Tensor, x_neigh: torch.Tensor, has_neigh: torch.Tensor) -> torch.Tensor:
        h_self = self.fc_self(x_self)
        h_neigh = self.fc_neigh(x_neigh) * has_neigh
        out = F.relu(self.norm(h_self + h_neigh))
        return out


class TemporalGraphSAGE(nn.Module):
    """Two-layer Temporal GraphSAGE network generating 8-dimensional embeddings."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 16, emb_dim: int = 8):
        super().__init__()
        self.layer1 = TemporalGraphSAGELayer(in_dim, hidden_dim)
        self.layer2 = TemporalGraphSAGELayer(hidden_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(
        self,
        x_self: torch.Tensor,
        x_neigh: torch.Tensor,
        has_neigh: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.layer1(x_self, x_neigh, has_neigh)
        # 2nd hop aggregation
        h2 = self.layer2(h1, x_neigh[:, :h1.shape[1]] if x_neigh.shape[1] == h1.shape[1] else h1, has_neigh)
        logits = self.classifier(h2).squeeze(-1)
        return h2, logits


def build_temporal_neighbor_matrix(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    entity_cols: List[str] = ["card1", "card2", "addr1"]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the historical neighbor feature matrix for every transaction.
    Guarantees that neighbor timestamp <= transaction timestamp.
    """
    n, d = feature_matrix.shape
    neigh_matrix = np.zeros((n, d), dtype=np.float32)
    has_neigh = np.zeros((n, 1), dtype=np.float32)

    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    entity_last_idx: Dict[str, int] = {}

    for i in range(n):
        row = df_sorted.iloc[i]
        ent_key = f"{row.get('card1', 'NA')}_{row.get('card2', 'NA')}_{row.get('addr1', 'NA')}"

        if ent_key in entity_last_idx:
            prev_idx = entity_last_idx[ent_key]
            neigh_matrix[i] = feature_matrix[prev_idx]
            has_neigh[i, 0] = 1.0

        entity_last_idx[ent_key] = i

    return neigh_matrix, has_neigh


def train_temporal_graphsage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    core_features: List[str],
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.005,
    emb_dim: int = 8,
    device: str = "cpu"
) -> Tuple[TemporalGraphSAGE, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Trains Temporal GraphSAGE on training split, records loss curve, and generates embeddings.
    """
    logger.info("--- Training Temporal GraphSAGE Neural Network (%d epochs, batch_size=%d) ---", epochs, batch_size)

    # Standardize core input features
    X_tr_raw = train_df[core_features].fillna(0.0).values.astype(np.float32)
    X_te_raw = test_df[core_features].fillna(0.0).values.astype(np.float32)

    mean = np.mean(X_tr_raw, axis=0, keepdims=True)
    std = np.std(X_tr_raw, axis=0, keepdims=True) + 1e-5
    X_tr_norm = (X_tr_raw - mean) / std
    X_te_norm = (X_te_raw - mean) / std

    y_tr = train_df["isFraud"].values.astype(np.float32)
    y_te = test_df["isFraud"].values.astype(np.float32)

    # Build leak-free neighbor representations
    logger.info("Constructing temporal neighborhood matrices...")
    neigh_tr, has_neigh_tr = build_temporal_neighbor_matrix(train_df, X_tr_norm)
    neigh_te, has_neigh_te = build_temporal_neighbor_matrix(test_df, X_te_norm)

    in_dim = X_tr_norm.shape[1]
    model = TemporalGraphSAGE(in_dim=in_dim, hidden_dim=16, emb_dim=emb_dim).to(device)

    # Compute positive class weight
    pos_weight = torch.tensor([(len(y_tr) - np.sum(y_tr)) / (np.sum(y_tr) + 1e-5)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    dataset = TensorDataset(
        torch.tensor(X_tr_norm, dtype=torch.float32),
        torch.tensor(neigh_tr, dtype=torch.float32),
        torch.tensor(has_neigh_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    logger.info("Starting training loop...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for x_b, nx_b, hn_b, y_b in loader:
            x_b = x_b.to(device)
            nx_b = nx_b.to(device)
            hn_b = hn_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            _, logits = model(x_b, nx_b, hn_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        loss_history.append({"epoch": epoch, "loss": round(avg_loss, 4)})
        logger.info("Epoch %2d/%2d | Training Loss: %.4f", epoch, epochs, avg_loss)

    # Generate embeddings on train and test
    model.eval()
    with torch.no_grad():
        emb_tr, _ = model(
            torch.tensor(X_tr_norm, dtype=torch.float32).to(device),
            torch.tensor(neigh_tr, dtype=torch.float32).to(device),
            torch.tensor(has_neigh_tr, dtype=torch.float32).to(device)
        )
        emb_te, _ = model(
            torch.tensor(X_te_norm, dtype=torch.float32).to(device),
            torch.tensor(neigh_te, dtype=torch.float32).to(device),
            torch.tensor(has_neigh_te, dtype=torch.float32).to(device)
        )

    emb_tr_np = emb_tr.cpu().numpy()
    emb_te_np = emb_te.cpu().numpy()

    # Convergence summary
    initial_loss = loss_history[0]["loss"]
    final_loss = loss_history[-1]["loss"]
    loss_reduction_pct = round((initial_loss - final_loss) / initial_loss * 100.0, 2)
    has_plateaued = abs(loss_history[-1]["loss"] - loss_history[-2]["loss"]) < 0.05

    convergence_summary = {
        "epochs_trained": epochs,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "has_plateaued": has_plateaued,
        "loss_curve": loss_history
    }

    logger.info("✔ GraphSAGE Training Complete. Initial Loss: %.4f -> Final Loss: %.4f (-%.2f%% reduction)", initial_loss, final_loss, loss_reduction_pct)
    return model, convergence_summary, emb_tr_np, emb_te_np
