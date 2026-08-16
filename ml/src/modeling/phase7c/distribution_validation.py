"""
AtmosIQ Phase 7C: Univariate Distributional Fidelity Validator (Workstream A).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp, skew, kurtosis


class UnivariateDistributionValidator:
    """Evaluates marginal distributional fidelity across all 35 prediction-safe features."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def validate_distributions(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        results = []
        eval_features = self.feature_registry + ["pm25"]

        for feat in eval_features:
            if feat not in df_real_train.columns or feat not in df_synthetic.columns:
                continue

            r_vals = df_real_train[feat].dropna().values
            s_vals = df_synthetic[feat].dropna().values

            # Summary stats Real
            r_mean, r_std = float(np.mean(r_vals)), float(np.std(r_vals))
            r_med = float(np.median(r_vals))
            r_min, r_max = float(np.min(r_vals)), float(np.max(r_vals))
            r_q25, r_q75 = float(np.percentile(r_vals, 25)), float(np.percentile(r_vals, 75))
            r_iqr = r_q75 - r_q25
            r_p5, r_p10 = float(np.percentile(r_vals, 5)), float(np.percentile(r_vals, 10))
            r_p90, r_p95, r_p99 = float(np.percentile(r_vals, 90)), float(np.percentile(r_vals, 95)), float(np.percentile(r_vals, 99))
            r_skew, r_kurt = float(skew(r_vals)), float(kurtosis(r_vals))

            # Summary stats Synth
            s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals))
            s_med = float(np.median(s_vals))
            s_min, s_max = float(np.min(s_vals)), float(np.max(s_vals))
            s_q25, s_q75 = float(np.percentile(s_vals, 25)), float(np.percentile(s_vals, 75))
            s_iqr = s_q75 - s_q25
            s_p5, s_p10 = float(np.percentile(s_vals, 5)), float(np.percentile(s_vals, 10))
            s_p90, s_p95, s_p99 = float(np.percentile(s_vals, 90)), float(np.percentile(s_vals, 95)), float(np.percentile(s_vals, 99))
            s_skew, s_kurt = float(skew(s_vals)), float(kurtosis(s_vals))

            # Statistical comparisons
            if len(np.unique(r_vals)) <= 2:
                scale = 1.0
            else:
                scale = max(r_std, 1.0)
            w1_norm = float(wasserstein_distance(r_vals / scale, s_vals / scale))
            ks_res = ks_2samp(r_vals, s_vals)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)
            smd = float(abs(r_mean - s_mean) / scale)
            q95_err = float(abs(r_p95 - s_p95))

            # Classification
            if w1_norm <= 0.08 and ks_stat <= 0.15:
                tier = "EXCELLENT"
            elif w1_norm <= 0.15 and ks_stat <= 0.25:
                tier = "ACCEPTABLE"
            elif w1_norm <= 0.25:
                tier = "WARNING"
            else:
                tier = "FAIL"

            results.append({
                "feature_name": feat,
                "tier": tier,
                "normalized_w1_distance": w1_norm,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "standardized_mean_diff": smd,
                "q95_error": q95_err,
                "real_mean": r_mean,
                "synth_mean": s_mean,
                "real_std": r_std,
                "synth_std": s_std,
                "real_median": r_med,
                "synth_median": s_med,
                "real_iqr": r_iqr,
                "synth_iqr": s_iqr,
                "real_skew": r_skew,
                "synth_skew": s_skew,
                "real_kurt": r_kurt,
                "synth_kurt": s_kurt,
                "real_p95": r_p95,
                "synth_p95": s_p95,
                "real_p99": r_p99,
                "synth_p99": s_p99,
            })

        df_dist = pd.DataFrame(results)
        mean_w1 = float(df_dist["normalized_w1_distance"].mean())
        mean_ks = float(df_dist["ks_statistic"].mean())
        pass_rate = float((df_dist["tier"].isin(["EXCELLENT", "ACCEPTABLE"])).mean() * 100.0)

        summary = {
            "mean_normalized_w1": mean_w1,
            "mean_ks_stat": mean_ks,
            "distribution_pass_rate_pct": pass_rate,
            "overall_status": "PASS" if mean_w1 <= 0.15 and pass_rate >= 85.0 else "WARNING",
        }

        return df_dist, summary
