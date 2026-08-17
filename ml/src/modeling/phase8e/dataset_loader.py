"""
AtmosIQ Phase 8E: Temporal Data Loader & Sequence Preprocessing Engine.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class Phase8ETemporalDataLoader:
    """Constructs temporal sequences with strict normalization isolation fitted only on 2020-2021 historical data."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False

    def fit_scaler(self, df_real_train: pd.DataFrame):
        """Fits standard scaler exclusively on real 2020-2021 historical training data."""
        common = [f for f in self.feature_registry if f in df_real_train.columns]
        X = df_real_train[common].values
        self.scaler.fit(X)
        self.is_scaler_fitted = True
        logger.info(f"Temporal scaler successfully fitted on {len(df_real_train)} historical observations across {len(common)} features.")

    def create_sequences(
        self,
        df: pd.DataFrame,
        window_size: int = 14,
        is_synthetic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates (N, window_size, d) feature sequences and (N,) target arrays."""
        if not self.is_scaler_fitted:
            raise RuntimeError("Scaler must be fitted on 2020-2021 historical data before creating sequences.")

        common = [f for f in self.feature_registry if f in df.columns]

        X_list, y_list = [], []

        if is_synthetic or "trajectory_id" in df.columns:
            # Process each trajectory independently to preserve temporal boundaries
            for _, df_t in df.groupby("trajectory_id"):
                if len(df_t) <= window_size:
                    continue
                feat_vals = self.scaler.transform(df_t[common].values)
                targets = df_t["pm25"].values
                for i in range(len(df_t) - window_size):
                    X_list.append(feat_vals[i : i + window_size])
                    y_list.append(targets[i + window_size])
        else:
            # Continuous chronological real series
            feat_vals = self.scaler.transform(df[common].values)
            targets = df["pm25"].values
            for i in range(len(df) - window_size):
                X_list.append(feat_vals[i : i + window_size])
                y_list.append(targets[i + window_size])

        if not X_list:
            return np.empty((0, window_size, len(common))), np.empty((0,))

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def build_augmented_training_set(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: Optional[pd.DataFrame],
        augmentation_ratio: float = 0.25,
        window_size: int = 14,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combines real sequences with synthetic sequences at the declared augmentation ratio."""
        X_real, y_real = self.create_sequences(df_real_train, window_size=window_size, is_synthetic=False)

        if df_synthetic is None or augmentation_ratio <= 0.0 or len(df_synthetic) == 0:
            return X_real, y_real

        X_synth_all, y_synth_all = self.create_sequences(df_synthetic, window_size=window_size, is_synthetic=True)
        n_synth_desired = int(len(X_real) * augmentation_ratio)

        if len(X_synth_all) > 0 and n_synth_desired > 0:
            np.random.seed(seed)
            indices = np.random.choice(len(X_synth_all), size=min(n_synth_desired, len(X_synth_all)), replace=False)
            X_synth = X_synth_all[indices]
            y_synth = y_synth_all[indices]
            X_combined = np.vstack([X_real, X_synth])
            y_combined = np.concatenate([y_real, y_synth])
            return X_combined, y_combined

        return X_real, y_real
