"""
AtmosIQ Phase 8B: Out-of-Distribution (OOD) Scale Monitor.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class OODScaleMonitor:
    """Monitors OOD support and feature space dispersion across scaling batches."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_dev = None
        self.baseline_p95_distance = 1.0

    def fit(self, df_real_dev: pd.DataFrame):
        """Fits baseline density support on historical development observations (2020-2021)."""
        common = [f for f in self.feature_registry if f in df_real_dev.columns]
        X_real = df_real_dev[common].values
        X_scaled = self.scaler.fit_transform(X_real)

        self.nn_dev = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
        self.nn_dev.fit(X_scaled)
        dists, _ = self.nn_dev.kneighbors(X_scaled)
        self.baseline_p95_distance = float(np.percentile(dists[:, 1], 95))

    def evaluate_batch_ood(self, df_batch: pd.DataFrame, batch_id: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Evaluates OOD metrics for an entire synthetic batch."""
        if self.nn_dev is None:
            raise RuntimeError("OODScaleMonitor must be fit on development data.")

        common = [f for f in self.feature_registry if f in df_batch.columns]
        X_batch = df_batch[common].values
        X_scaled = self.scaler.transform(X_batch)

        dists, _ = self.nn_dev.kneighbors(X_scaled, n_neighbors=1)
        nn_dists = dists[:, 0]

        # Categorize density support
        in_dist_mask = (nn_dists <= self.baseline_p95_distance)
        expanded_mask = (nn_dists > self.baseline_p95_distance) & (nn_dists <= self.baseline_p95_distance * 1.5)
        outlier_mask = (nn_dists > self.baseline_p95_distance * 1.5)

        total_obs = len(df_batch)
        in_dist_pct = float(in_dist_mask.sum() / total_obs * 100.0) if total_obs > 0 else 0.0
        expanded_pct = float(expanded_mask.sum() / total_obs * 100.0) if total_obs > 0 else 0.0
        outlier_pct = float(outlier_mask.sum() / total_obs * 100.0) if total_obs > 0 else 0.0

        summary = {
            "batch_id": batch_id,
            "observation_count": total_obs,
            "median_nn_distance": float(np.median(nn_dists)),
            "p95_nn_distance": float(np.percentile(nn_dists, 95)),
            "in_distribution_pct": in_dist_pct,
            "expanded_support_pct": expanded_pct,
            "outlier_pct": outlier_pct,
            "ood_status": "PASS" if outlier_pct <= 50.0 else "WARNING",
        }

        df_annotated = df_batch.copy()
        df_annotated["ood_distance"] = nn_dists
        df_annotated["ood_category"] = np.where(in_dist_mask, "in_distribution", np.where(expanded_mask, "expanded_support", "outlier"))

        return summary, df_annotated
