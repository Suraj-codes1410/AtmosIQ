"""
AtmosIQ Phase 8G: Temporal Sequence Construction & Transformation Engine.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

from .policy_engine import Phase8GAugmentationPolicyEngine, AugmentationPolicyViolation

logger = logging.getLogger(__name__)


class Phase8GSequenceBuilder:
    """Builds leakage-safe, trajectory-bounded temporal sequences and feature tensors for Phase 9."""

    def __init__(self, feature_registry: List[str], target_variable: str = "pm25"):
        self.feature_registry = list(feature_registry)
        self.target_variable = target_variable
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        self.policy_engine = Phase8GAugmentationPolicyEngine()

    def fit_scaler(self, df_real_train: pd.DataFrame):
        """Fits StandardScaler strictly on 2020-2021 historical real training data."""
        common = [f for f in self.feature_registry if f in df_real_train.columns]
        X = df_real_train[common].values
        self.scaler.fit(X)
        self.is_scaler_fitted = True
        logger.info(f"Phase 8G sequence scaler fitted on {len(df_real_train)} historical rows across {len(common)} features.")

    def create_sequences_from_trajectories(
        self,
        df: pd.DataFrame,
        window_size: int = 14,
        is_synthetic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Constructs temporal sequences (N, W, D) and aligned targets (N,) respecting trajectory boundaries."""
        if not self.is_scaler_fitted:
            raise RuntimeError("Scaler must be fitted exclusively on 2020-2021 historical data before building sequences.")

        common = [f for f in self.feature_registry if f in df.columns]
        X_seqs, y_targets, provenance_list = [], [], []

        if is_synthetic or "trajectory_id" in df.columns:
            for traj_id, df_t in df.groupby("trajectory_id"):
                if len(df_t) <= window_size:
                    continue
                # Normalize features
                feat_scaled = self.scaler.transform(df_t[common].values)
                targets = df_t[self.target_variable].values
                origin = df_t["data_origin"].iloc[0] if "data_origin" in df_t else ("synthetic" if is_synthetic else "real")
                partition = df_t["source_partition"].iloc[0] if "source_partition" in df_t else "2020-2021"

                for i in range(len(df_t) - window_size):
                    X_seqs.append(feat_scaled[i : i + window_size])
                    y_targets.append(targets[i + window_size])
                    provenance_list.append({
                        "trajectory_id": str(traj_id),
                        "window_slice_idx": i,
                        "data_origin": origin,
                        "source_partition": partition,
                        "window_size": window_size,
                    })
        else:
            # Continuous historical series
            feat_scaled = self.scaler.transform(df[common].values)
            targets = df[self.target_variable].values
            for i in range(len(df) - window_size):
                X_seqs.append(feat_scaled[i : i + window_size])
                y_targets.append(targets[i + window_size])
                provenance_list.append({
                    "trajectory_id": "REAL_HISTORICAL_2020_2021",
                    "window_slice_idx": i,
                    "data_origin": "real",
                    "source_partition": "2020-2021",
                    "window_size": window_size,
                })

        if not X_seqs:
            return np.empty((0, window_size, len(common)), dtype=np.float32), np.empty((0,), dtype=np.float32), []

        return (
            np.array(X_seqs, dtype=np.float32),
            np.array(y_targets, dtype=np.float32),
            provenance_list,
        )

    def assemble_integrated_dataset(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: Optional[pd.DataFrame],
        augmentation_ratio: float = 0.25,
        window_size: int = 14,
        seed: int = 42,
        is_stress_test: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """Assembles integrated dataset according to governance policy tiers."""
        # 1. Policy validation
        policy_res = self.policy_engine.validate_augmentation_request(augmentation_ratio, is_stress_test=is_stress_test)

        # 2. Extract real sequences
        X_real, y_real, prov_real = self.create_sequences_from_trajectories(df_real_train, window_size=window_size, is_synthetic=False)

        if df_synthetic is None or augmentation_ratio <= 0.0 or len(df_synthetic) == 0:
            df_prov = pd.DataFrame(prov_real)
            meta = {
                "augmentation_ratio": augmentation_ratio,
                "tier": policy_res["tier"],
                "total_sequences": len(X_real),
                "real_sequences": len(X_real),
                "synthetic_sequences": 0,
                "feature_dim": X_real.shape[2],
                "window_size": window_size,
            }
            return X_real, y_real, df_prov, meta

        # 3. Extract synthetic sequences
        X_synth_all, y_synth_all, prov_synth_all = self.create_sequences_from_trajectories(df_synthetic, window_size=window_size, is_synthetic=True)
        n_synth_desired = int(len(X_real) * augmentation_ratio)

        np.random.seed(seed)
        indices = np.random.choice(len(X_synth_all), size=min(n_synth_desired, len(X_synth_all)), replace=False)
        X_synth = X_synth_all[indices]
        y_synth = y_synth_all[indices]
        prov_synth = [prov_synth_all[idx] for idx in indices]

        X_combined = np.vstack([X_real, X_synth])
        y_combined = np.concatenate([y_real, y_synth])
        df_prov = pd.DataFrame(prov_real + prov_synth)

        meta = {
            "augmentation_ratio": augmentation_ratio,
            "tier": policy_res["tier"],
            "total_sequences": len(X_combined),
            "real_sequences": len(X_real),
            "synthetic_sequences": len(X_synth),
            "feature_dim": X_combined.shape[2],
            "window_size": window_size,
        }

        return X_combined, y_combined, df_prov, meta
