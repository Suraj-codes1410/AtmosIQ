"""
AtmosIQ Phase 7C: Multivariate Dependency Fidelity Validator (Workstream B).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import norm


class MultivariateDependencyValidator:
    """Evaluates cross-feature correlation and covariance preservation."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def validate_multivariate_dependencies(
        self,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray]:
        common = [f for f in self.feature_registry if f in df_real.columns and f in df_synthetic.columns]
        
        # Pearson and Spearman
        corr_r_pearson = df_real[common].corr(method="pearson").fillna(0.0).values
        corr_s_pearson = df_synthetic[common].corr(method="pearson").fillna(0.0).values

        corr_r_spearman = df_real[common].corr(method="spearman").fillna(0.0).values
        corr_s_spearman = df_synthetic[common].corr(method="spearman").fillna(0.0).values

        d = len(common)
        frob_pearson = float(norm(corr_r_pearson - corr_s_pearson, "fro") / d)
        frob_spearman = float(norm(corr_r_spearman - corr_s_spearman, "fro") / d)

        abs_diff_pearson = np.abs(corr_r_pearson - corr_s_pearson)
        mean_abs_diff = float(np.mean(abs_diff_pearson))
        max_abs_diff = float(np.max(abs_diff_pearson))

        # Key Atmospheric Relationship Checks
        key_pairs = [
            ("pm25_lag_1d", "pm25_roll_mean_3d"),
            ("pm25_roll_mean_3d", "pm25_roll_mean_7d"),
            ("wind_speed_kmh_lag_1d", "ventilation_index_1d"),
            ("pblh_1d", "ventilation_index_1d"),
            ("rainfall_1d", "washout_index_3d"),
            ("temperature_c_lag_1d", "temperature_c_roll_mean_3d"),
            ("humidity_pct_lag_1d", "humidity_pct_roll_mean_3d"),
        ]

        pair_records = []
        for v1, v2 in key_pairs:
            if v1 in df_real.columns and v2 in df_real.columns and v1 in df_synthetic.columns and v2 in df_synthetic.columns:
                r_corr = float(df_real[[v1, v2]].corr().iloc[0, 1])
                s_corr = float(df_synthetic[[v1, v2]].corr().iloc[0, 1])
                delta = abs(r_corr - s_corr)
                pair_records.append({
                    "feature_1": v1,
                    "feature_2": v2,
                    "real_correlation": r_corr,
                    "synth_correlation": s_corr,
                    "absolute_delta": delta,
                    "status": "PASS" if delta <= 0.20 else "WARNING",
                })

        df_pairs = pd.DataFrame(pair_records)

        summary = {
            "pearson_frobenius_distance": frob_pearson,
            "spearman_frobenius_distance": frob_spearman,
            "mean_absolute_correlation_diff": mean_abs_diff,
            "max_absolute_correlation_diff": max_abs_diff,
            "overall_status": "PASS" if frob_pearson <= 0.20 else "WARNING",
        }

        return df_pairs, summary, corr_r_pearson, corr_s_pearson
