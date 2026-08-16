"""
AtmosIQ Phase 8A: Memorization & Duplicate Protection Engine.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MemorizationScreen:
    """Provides Fast Screen and Full Audit checks for historical memorization."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_dev = None
        self.dev_feature_matrix = None

    def fit(self, df_real_dev: pd.DataFrame):
        """Fits baseline against historical development observations (2020-2021)."""
        common = [f for f in self.feature_registry if f in df_real_dev.columns]
        self.dev_feature_matrix = df_real_dev[common].values
        X_scaled = self.scaler.fit_transform(self.dev_feature_matrix)
        self.nn_dev = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        self.nn_dev.fit(X_scaled)

    def screen_trajectory(
        self,
        df_traj: pd.DataFrame,
        trajectory_id: str,
        mode: str = "FAST_SCREEN"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Screens trajectory for exact and near-duplicates against real historical records.
        Returns (is_passed, audit_report).
        """
        if self.nn_dev is None:
            raise RuntimeError("MemorizationScreen must be fit on development data before screening.")

        common = [f for f in self.feature_registry if f in df_traj.columns]
        X_traj = df_traj[common].values
        X_scaled = self.scaler.transform(X_traj)

        dists, idxs = self.nn_dev.kneighbors(X_scaled)
        dists_vec = dists[:, 0]

        exact_dups = int((dists_vec == 0.0).sum())
        near_dups = int((dists_vec < 0.05).sum())

        passed = (exact_dups == 0 and near_dups == 0)

        report = {
            "trajectory_id": trajectory_id,
            "mode": mode,
            "passed": passed,
            "exact_duplicates": exact_dups,
            "near_duplicates": near_dups,
            "min_distance_to_real": float(np.min(dists_vec)),
            "mean_distance_to_real": float(np.mean(dists_vec)),
        }

        return passed, report
