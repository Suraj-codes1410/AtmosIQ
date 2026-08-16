"""
AtmosIQ Phase 7C: Synthetic Out-of-Distribution & Artifact Auditor (Workstream J).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class SyntheticOODAuditor:
    """Audits whether synthetic observations occupy realistic regions of historical feature space."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def audit_ood_artifacts(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        common = [f for f in self.feature_registry if f in df_real_train.columns and f in df_synthetic.columns]

        scaler = StandardScaler()
        X_real_scaled = scaler.fit_transform(df_real_train[common].values)
        X_synth_scaled = scaler.transform(df_synthetic[common].values)

        # 1. Real-to-Real nearest neighbor distances (baseline)
        nn_real = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1).fit(X_real_scaled)
        dists_r2r, _ = nn_real.kneighbors(X_real_scaled)
        baseline_nn_dist = dists_r2r[:, 1]  # 2nd neighbor (1st is self)

        # 2. Synthetic-to-Real nearest neighbor distances
        nn_synth_to_real = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(X_real_scaled)
        dists_s2r, idxs_s2r = nn_synth_to_real.kneighbors(X_synth_scaled)
        s2r_dist = dists_s2r[:, 0]

        r_median_nn = float(np.median(baseline_nn_dist))
        r_p95_nn = float(np.percentile(baseline_nn_dist, 95))

        s_median_nn = float(np.median(s2r_dist))
        s_p95_nn = float(np.percentile(s2r_dist, 95))

        # Classify each synthetic observation
        # Near OOD if > p95 of real-to-real
        ood_flags = (s2r_dist > r_p95_nn * 1.5).astype(int)
        ood_pct = float(ood_flags.mean() * 100.0)

        df_audit = pd.DataFrame({
            "trajectory_id": df_synthetic["trajectory_id"],
            "step_idx": df_synthetic["step_idx"],
            "synthetic_date": df_synthetic["synthetic_date"],
            "nearest_real_distance": s2r_dist,
            "is_outlier_flag": ood_flags,
        })

        summary = {
            "real_baseline_median_nn_dist": r_median_nn,
            "real_baseline_p95_nn_dist": r_p95_nn,
            "synthetic_median_nn_dist": s_median_nn,
            "synthetic_p95_nn_dist": s_p95_nn,
            "synthetic_outlier_count": int(ood_flags.sum()),
            "synthetic_outlier_pct": ood_pct,
            "ood_status": "PASS" if ood_pct <= 10.0 else "WARNING",
        }

        return df_audit, summary
