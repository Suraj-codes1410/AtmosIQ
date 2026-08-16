"""
AtmosIQ Phase 7C: Memorization and Duplication Auditor (Workstream K).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class MemorizationAuditor:
    """Audits whether the generator reproduced or memorized historical records."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def audit_memorization(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        common = [f for f in self.feature_registry if f in df_real_train.columns and f in df_synthetic.columns]

        scaler = StandardScaler()
        X_real_scaled = scaler.fit_transform(df_real_train[common].values)
        X_synth_scaled = scaler.transform(df_synthetic[common].values)

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(X_real_scaled)
        dists, indices = nn.kneighbors(X_synth_scaled)
        dists_vec = dists[:, 0]

        # Exact duplicate if distance == 0.0
        exact_dups = int((dists_vec == 0.0).sum())
        # Near duplicate if distance < 0.05
        near_dups = int((dists_vec < 0.05).sum())

        df_records = pd.DataFrame({
            "synthetic_date": df_synthetic["synthetic_date"],
            "trajectory_id": df_synthetic["trajectory_id"],
            "nearest_real_index": indices[:, 0],
            "euclidean_distance_to_nearest_real": dists_vec,
            "is_exact_duplicate": (dists_vec == 0.0).astype(int),
            "is_near_duplicate": (dists_vec < 0.05).astype(int),
        })

        summary = {
            "total_synthetic_records": len(df_synthetic),
            "exact_duplicate_count": exact_dups,
            "near_duplicate_count": near_dups,
            "min_distance_to_real": float(np.min(dists_vec)),
            "mean_distance_to_real": float(np.mean(dists_vec)),
            "memorization_status": "PASS" if exact_dups == 0 and near_dups == 0 else "FAIL_MEMORIZATION_DETECTED",
        }

        return df_records, summary
