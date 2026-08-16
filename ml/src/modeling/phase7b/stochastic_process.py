"""
AtmosIQ Phase 7B: Correlated Stochastic Innovation Sampler.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class CorrelatedInnovationSampler:
    """Samples correlated stochastic innovations preserving regime-conditioned covariance."""

    def __init__(self, random_state: np.random.RandomState):
        self.rng = random_state
        self.regime_covariances: Dict[str, np.ndarray] = {}
        self.regime_means: Dict[str, np.ndarray] = {}
        self.variables = [
            "pm25_delta", "temperature_c", "humidity_pct",
            "wind_speed_kmh", "pblh_1d", "rainfall_1d"
        ]

    def fit_from_training_data(self, df_train: pd.DataFrame):
        df = df_train.copy()
        if "pollution_regime" not in df.columns:
            def classify_regime(pm):
                if pm < 60.0: return "Low"
                if pm < 120.0: return "Moderate"
                if pm < 250.0: return "High"
                return "Extreme"
            df["pollution_regime"] = df["pm25"].apply(classify_regime)

        df["pm25_delta"] = df["pm25"] - df["pm25"].shift(1).fillna(df["pm25"])

        regimes = ["Low", "Moderate", "High", "Extreme"]
        for r in regimes:
            sub = df[df["pollution_regime"] == r]
            if len(sub) > 5:
                sub_vars = sub[["pm25_delta", "temperature_c", "humidity_pct", "wind_speed_kmh", "pblh_1d", "rainfall_1d"]].dropna()
                mean_vec = sub_vars.mean().values
                cov_mat = np.cov(sub_vars.values, rowvar=False)
                # Add small ridge for positive definiteness
                cov_mat += np.eye(len(self.variables)) * 1e-4
            else:
                mean_vec = np.zeros(len(self.variables))
                cov_mat = np.eye(len(self.variables))

            self.regime_means[r] = mean_vec
            self.regime_covariances[r] = cov_mat

    def sample_innovation(self, regime: str) -> Dict[str, float]:
        mean_vec = self.regime_means.get(regime, np.zeros(len(self.variables)))
        cov_mat = self.regime_covariances.get(regime, np.eye(len(self.variables)))
        
        sample = self.rng.multivariate_normal(mean_vec, cov_mat)
        return {var: float(sample[i]) for i, var in enumerate(self.variables)}
