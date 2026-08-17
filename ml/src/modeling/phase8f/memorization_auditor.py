"""
AtmosIQ Phase 8F: Memorization & Duplicate Auditing Engine.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class Phase8FMemorizationAuditor:
    """Audits synthetic observations against historical training data for exact or near-duplicate memorization."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_dev = None

    def fit_reference(self, df_real_train: pd.DataFrame):
        common = [f for f in self.feature_registry if f in df_real_train.columns]
        X_real = df_real_train[common].values
        X_scaled = self.scaler.fit_transform(X_real)
        self.nn_dev = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        self.nn_dev.fit(X_scaled)

    def audit_memorization(self, df_8c: pd.DataFrame, df_8d: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        checks = []
        common = [f for f in self.feature_registry if f in df_8c.columns and f in df_8d.columns]

        for name, df in [("AtmosIQ_Synthetic_Production_v1.0.0", df_8c), ("AtmosIQ_Synthetic_Calibrated_v0.1.0", df_8d)]:
            X_synth = df[common].values
            X_scaled = self.scaler.transform(X_synth)
            dists, _ = self.nn_dev.kneighbors(X_scaled)
            dist_vec = dists[:, 0]

            exact_dups = int((dist_vec <= 1e-6).sum())
            near_dups = int(((dist_vec > 1e-6) & (dist_vec < 0.05)).sum())
            min_dist = float(np.min(dist_vec))
            mean_dist = float(np.mean(dist_vec))

            checks.append({
                "corpus": name,
                "check": "Exact Historical Duplicates (d <= 1e-6)",
                "violations": exact_dups,
                "status": "PASS" if exact_dups == 0 else "FAIL",
                "details": f"Zero exact copies found (min distance: {min_dist:.4f})",
            })
            checks.append({
                "corpus": name,
                "check": "Near-Duplicate Memorization (d < 0.05)",
                "violations": near_dups,
                "status": "PASS" if near_dups == 0 else "FAIL",
                "details": f"Zero near-copies found (mean distance: {mean_dist:.4f})",
            })

        df_aud = pd.DataFrame(checks)
        all_passed = bool((df_aud["status"] == "PASS").all())
        return all_passed, df_aud
