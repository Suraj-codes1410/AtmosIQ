"""
AtmosIQ Phase 8B: Memorization & Duplicate Scale Auditor.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MemorizationScaleAuditor:
    """Audits synthetic batches for historical memorization and intra-corpus duplication."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_dev = None

    def fit(self, df_real_dev: pd.DataFrame):
        """Fits against real development dataset."""
        common = [f for f in self.feature_registry if f in df_real_dev.columns]
        X_real = df_real_dev[common].values
        X_scaled = self.scaler.fit_transform(X_real)
        self.nn_dev = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        self.nn_dev.fit(X_scaled)

    def audit_batch(self, df_batch: pd.DataFrame, batch_id: str) -> Dict[str, Any]:
        """Audits a batch of trajectories against historical development data."""
        if self.nn_dev is None:
            raise RuntimeError("MemorizationScaleAuditor must be fit on development data.")

        common = [f for f in self.feature_registry if f in df_batch.columns]
        X_batch = df_batch[common].values
        X_scaled = self.scaler.transform(X_batch)

        dists, idxs = self.nn_dev.kneighbors(X_scaled)
        dists_vec = dists[:, 0]

        exact_dups = int((dists_vec <= 1e-6).sum())
        near_dups = int(((dists_vec > 1e-6) & (dists_vec < 0.05)).sum())

        return {
            "batch_id": batch_id,
            "total_observations": len(df_batch),
            "exact_duplicate_count": exact_dups,
            "near_duplicate_count": near_dups,
            "min_distance_to_real": float(np.min(dists_vec)) if len(dists_vec) > 0 else 0.0,
            "mean_distance_to_real": float(np.mean(dists_vec)) if len(dists_vec) > 0 else 0.0,
            "memorization_status": "PASS" if exact_dups == 0 and near_dups == 0 else "FAIL_MEMORIZATION_DETECTED",
        }
