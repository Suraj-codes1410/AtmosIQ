"""
AtmosIQ Phase 8A: OOD Support & Density Feature Space Auditor.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class OODSupportScorer:
    """Computes feature-space density support and OOD flags against development manifold."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_model = None
        self.baseline_p95_distance = 1.0

    def fit(self, df_real_dev: pd.DataFrame):
        """Fits density baseline on historical development data (2020-2021)."""
        common = [f for f in self.feature_registry if f in df_real_dev.columns]
        X_real = df_real_dev[common].values
        X_scaled = self.scaler.fit_transform(X_real)

        # Fit nearest neighbors
        self.nn_model = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
        self.nn_model.fit(X_scaled)
        dists, _ = self.nn_model.kneighbors(X_scaled)
        self.baseline_p95_distance = float(np.percentile(dists[:, 1], 95))

    def annotate_trajectory(self, df_traj: pd.DataFrame) -> pd.DataFrame:
        """Annotates synthetic trajectory with OOD distance and outlier metadata."""
        if self.nn_model is None:
            raise RuntimeError("OODSupportScorer must be fit on development data before annotation.")

        common = [f for f in self.feature_registry if f in df_traj.columns]
        X_traj = df_traj[common].values
        X_scaled = self.scaler.transform(X_traj)

        dists, _ = self.nn_model.kneighbors(X_scaled, n_neighbors=1)
        nn_dists = dists[:, 0]

        # Compute max absolute z-score per row
        max_zscores = np.max(np.abs(X_scaled), axis=1)

        # OOD threshold
        is_ood = (nn_dists > self.baseline_p95_distance * 1.5).astype(int)

        df_annotated = df_traj.copy()
        df_annotated["ood_distance"] = nn_dists
        df_annotated["max_feature_zscore"] = max_zscores
        df_annotated["is_ood_flag"] = is_ood

        return df_annotated
