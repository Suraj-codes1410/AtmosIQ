"""
AtmosIQ Phase 8B: Scale-Dependent Fidelity Monitoring Engine.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.linalg import norm


class FidelityScaleMonitor:
    """Evaluates univariate, multivariate, temporal, and extreme-tail fidelity for scaled batches."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def evaluate_batch_fidelity(
        self,
        df_real_dev: pd.DataFrame,
        df_batch: pd.DataFrame,
        batch_id: str
    ) -> Dict[str, Any]:
        common = [f for f in self.feature_registry if f in df_real_dev.columns and f in df_batch.columns]

        # 1. Univariate Wasserstein-1
        w1_list = []
        for feat in common + ["pm25"]:
            if feat not in df_real_dev.columns or feat not in df_batch.columns:
                continue
            r_vals = df_real_dev[feat].dropna().values
            b_vals = df_batch[feat].dropna().values

            if len(np.unique(r_vals)) <= 2:
                scale = 1.0
            else:
                scale = max(float(np.std(r_vals)), 1.0)

            w1_norm = float(wasserstein_distance(r_vals / scale, b_vals / scale))
            w1_list.append(w1_norm)

        mean_w1 = float(np.mean(w1_list)) if w1_list else 0.0

        # 2. Multivariate Correlation Frobenius Distance
        corr_r = df_real_dev[common].corr(method="pearson").fillna(0.0).values
        corr_b = df_batch[common].corr(method="pearson").fillna(0.0).values
        d = len(common)
        frob_dist = float(norm(corr_r - corr_b, "fro") / d) if d > 0 else 0.0

        # 3. Temporal ACF Error
        def compute_acf(s, max_lag=30):
            s = s - np.mean(s)
            var = np.var(s)
            if var == 0: return np.ones(max_lag)
            return [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in range(1, max_lag + 1)]

        acf_r = compute_acf(df_real_dev["pm25"].dropna().values, 30)
        acf_b = compute_acf(df_batch["pm25"].dropna().values, 30)

        acf_errors = [abs(acf_r[i] - acf_b[i]) for i in range(len(acf_r))]
        mean_acf_err_7 = float(np.mean(acf_errors[:7]))
        mean_acf_err_30 = float(np.mean(acf_errors))

        # 4. Extreme-Tail Coherence
        ext_250 = df_batch[df_batch["pm25"] >= 250.0]
        ext_count = len(ext_250)
        ext_pct = float(ext_count / len(df_batch) * 100.0) if len(df_batch) > 0 else 0.0

        if ext_count > 0:
            coherent = (ext_250["ventilation_index_1d"] <= 4500.0) & (ext_250["rainfall_1d"] <= 2.0)
            coherence_rate = float(coherent.mean() * 100.0)
        else:
            coherence_rate = 100.0

        # 5. Seasonal & Regime Distribution
        reg_counts = df_batch["pollution_regime"].value_counts(normalize=True).to_dict() if "pollution_regime" in df_batch else {}
        seas_counts = df_batch["season"].value_counts(normalize=True).to_dict() if "season" in df_batch else {}

        return {
            "batch_id": batch_id,
            "observation_count": len(df_batch),
            "mean_normalized_w1": mean_w1,
            "frobenius_correlation_distance": frob_dist,
            "mean_acf_error_lags_1_7": mean_acf_err_7,
            "mean_acf_error_lags_1_30": mean_acf_err_30,
            "extreme_250_count": ext_count,
            "extreme_250_pct": ext_pct,
            "extreme_coherence_rate_pct": coherence_rate,
            "regime_distribution": {k: float(v * 100.0) for k, v in reg_counts.items()},
            "season_distribution": {k: float(v * 100.0) for k, v in seas_counts.items()},
            "fidelity_status": "PASS" if mean_w1 <= 0.60 and frob_dist <= 0.30 else "WARNING",
        }
