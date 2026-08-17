"""
AtmosIQ Phase 8D: Multi-Objective Fidelity & Statistical Evaluator.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class MultiObjectiveFidelityEvaluator:
    """Evaluates comprehensive statistical, temporal, multivariate, and physical fidelity."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_dev = None
        self.baseline_p95_distance = 1.0

    def fit_reference(self, df_real_dev: pd.DataFrame):
        common = [f for f in self.feature_registry if f in df_real_dev.columns]
        X_real = df_real_dev[common].values
        X_scaled = self.scaler.fit_transform(X_real)
        self.nn_dev = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
        self.nn_dev.fit(X_scaled)
        dists, _ = self.nn_dev.kneighbors(X_scaled)
        self.baseline_p95_distance = float(np.percentile(dists[:, 1], 95))

    def evaluate_candidate(
        self,
        df_real_dev: pd.DataFrame,
        df_candidate: pd.DataFrame,
        candidate_id: str
    ) -> Dict[str, Any]:
        common = [f for f in self.feature_registry if f in df_real_dev.columns and f in df_candidate.columns]

        # 1. Univariate Normalized Wasserstein
        w1_list = []
        for feat in common + ["pm25"]:
            if feat not in df_real_dev.columns or feat not in df_candidate.columns:
                continue
            r_vals = df_real_dev[feat].dropna().values
            c_vals = df_candidate[feat].dropna().values
            scale = max(float(np.std(r_vals)), 1.0) if len(np.unique(r_vals)) > 2 else 1.0
            w1_norm = float(wasserstein_distance(r_vals / scale, c_vals / scale))
            w1_list.append(w1_norm)

        mean_w1 = float(np.mean(w1_list)) if w1_list else 0.0

        # 2. Multivariate Correlation Frobenius Distance
        corr_r = df_real_dev[common].corr(method="pearson").fillna(0.0).values
        corr_c = df_candidate[common].corr(method="pearson").fillna(0.0).values
        d = len(common)
        frob_dist = float(norm(corr_r - corr_c, "fro") / d) if d > 0 else 0.0

        # 3. Temporal ACF (Lags 1-30)
        def compute_acf(s, max_lag=30):
            s = s - np.mean(s)
            var = np.var(s)
            if var <= 1e-6: return np.ones(max_lag)
            return [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in range(1, max_lag + 1)]

        acf_r = compute_acf(df_real_dev["pm25"].dropna().values, 30)
        acf_c = compute_acf(df_candidate["pm25"].dropna().values, 30)
        acf_errors = [abs(acf_r[i] - acf_c[i]) for i in range(len(acf_r))]
        mean_acf_err_7 = float(np.mean(acf_errors[:7]))
        mean_acf_err_30 = float(np.mean(acf_errors))

        # 4. Seasonal & Regime Distribution
        reg_r = df_real_dev["pollution_regime"].value_counts(normalize=True).to_dict() if "pollution_regime" in df_real_dev else {}
        reg_c = df_candidate["pollution_regime"].value_counts(normalize=True).to_dict() if "pollution_regime" in df_candidate else {}
        reg_err = float(np.mean([abs(reg_r.get(k, 0.0) - reg_c.get(k, 0.0)) for k in ["Low", "Moderate", "High", "Extreme"]]))

        # 5. Extreme-Tail Coherence
        ext_250 = df_candidate[df_candidate["pm25"] >= 250.0]
        ext_count = len(ext_250)
        if ext_count > 0:
            coherent = (ext_250["ventilation_index_1d"] <= 4500.0) & (ext_250["rainfall_1d"] <= 2.0)
            coherence_rate = float(coherent.mean() * 100.0)
        else:
            coherence_rate = 100.0

        # 6. OOD Density Support
        X_cand = df_candidate[common].values
        X_cand_scaled = self.scaler.transform(X_cand)
        dists, _ = self.nn_dev.kneighbors(X_cand_scaled, n_neighbors=1)
        nn_dists = dists[:, 0]
        outlier_pct = float((nn_dists > self.baseline_p95_distance * 1.5).sum() / len(df_candidate) * 100.0) if len(df_candidate) > 0 else 0.0

        # 7. Physical Validity & Hydrodynamics
        ws_ms = df_candidate["wind_speed_kmh"] * (1000.0 / 3600.0)
        expected_vi = ws_ms * df_candidate["pblh_1d"]
        bad_vi = int((np.abs(df_candidate["ventilation_index_1d"] - expected_vi) > 1.0).sum())
        neg_pm = int((df_candidate["pm25"] < 0.0).sum())
        phys_valid_pct = 100.0 if (bad_vi == 0 and neg_pm == 0) else 0.0

        return {
            "candidate_id": candidate_id,
            "observations": len(df_candidate),
            "trajectories": df_candidate["trajectory_id"].nunique(),
            "mean_normalized_w1": mean_w1,
            "frobenius_correlation_distance": frob_dist,
            "mean_acf_error_lags_1_7": mean_acf_err_7,
            "mean_acf_error_lags_1_30": mean_acf_err_30,
            "regime_distribution_error": reg_err,
            "extreme_coherence_rate_pct": coherence_rate,
            "ood_outlier_pct": outlier_pct,
            "physical_validity_pct": phys_valid_pct,
            "hydrodynamic_violations": bad_vi,
        }
