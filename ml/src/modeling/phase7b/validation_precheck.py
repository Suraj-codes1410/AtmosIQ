"""
AtmosIQ Phase 7B: Validation Pre-Check Engine.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.linalg import norm

from .extreme_event import ExtremeEventGenerator


class ValidationPrecheckerPhase7B:
    """
    Computes preliminary distributional, correlation, temporal, and physical checks.
    """

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.extreme_evaluator = ExtremeEventGenerator()

    def run_precheck(self, df_real_train: pd.DataFrame, df_synthetic: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive statistical and physical pre-checks.
        """
        # 1. Physical constraint compliance
        neg_pm = int((df_synthetic["pm25"] < 0).sum())
        neg_ws = int((df_synthetic["wind_speed_kmh"] < 0).sum())
        neg_rain = int((df_synthetic["rainfall_1d"] < 0).sum())
        neg_pblh = int((df_synthetic["pblh_1d"] <= 0).sum())
        nan_count = int(df_synthetic[self.feature_registry + ["pm25"]].isna().sum().sum())

        physical_pass = (neg_pm == 0 and neg_ws == 0 and neg_rain == 0 and neg_pblh == 0 and nan_count == 0)

        # 2. Wasserstein-1 Distance on core variables
        w1_scores = {}
        ks_scores = {}
        eval_vars = ["pm25", "temperature_c_lag_1d", "humidity_pct_lag_1d", "wind_speed_kmh_lag_1d", "pblh_1d", "aod_550_1d", "ventilation_index_1d"]
        for var in eval_vars:
            if var in df_real_train.columns and var in df_synthetic.columns:
                real_v = df_real_train[var].dropna().values
                synth_v = df_synthetic[var].dropna().values
                # Normalize by real std
                scale = max(float(np.std(real_v)), 1e-4)
                w1 = float(wasserstein_distance(real_v / scale, synth_v / scale))
                ks_stat = float(ks_2samp(real_v, synth_v).statistic)
                w1_scores[var] = w1
                ks_scores[var] = ks_stat

        mean_w1 = float(np.mean(list(w1_scores.values())))
        mean_ks = float(np.mean(list(ks_scores.values())))

        # 3. Correlation Matrix Frobenius Distance
        common_features = [f for f in self.feature_registry if f in df_real_train.columns and f in df_synthetic.columns]
        corr_real = df_real_train[common_features].corr().fillna(0.0).values
        corr_synth = df_synthetic[common_features].corr().fillna(0.0).values
        
        # Normalized Frobenius distance = ||C_real - C_synth||_F / d
        d = len(common_features)
        frob_dist = float(norm(corr_real - corr_synth, 'fro') / d)

        # 4. Temporal Autocorrelation (ACF) Distance for PM2.5 Lags 1–7
        def compute_acf(series, max_lag=7):
            s = series - np.mean(series)
            var = np.var(series)
            if var == 0: return np.ones(max_lag)
            acf = [np.corrcoef(series[:-k], series[k:])[0, 1] for k in range(1, max_lag + 1)]
            return np.array(acf)

        acf_real = compute_acf(df_real_train["pm25"].dropna().values)
        acf_synth = compute_acf(df_synthetic["pm25"].dropna().values)
        mean_acf_error = float(np.mean(np.abs(acf_real - acf_synth)))

        # 5. Extreme Event Coherence
        extreme_res = self.extreme_evaluator.evaluate_extreme_coherence(df_synthetic)

        # 6. Regime Distributions
        def get_regime_dist(df):
            if "pollution_regime" not in df.columns:
                def classify_regime(pm):
                    if pm < 60.0: return "Low"
                    if pm < 120.0: return "Moderate"
                    if pm < 250.0: return "High"
                    return "Extreme"
                df_temp = df.copy()
                df_temp["pollution_regime"] = df_temp["pm25"].apply(classify_regime)
            else:
                df_temp = df
            counts = df_temp["pollution_regime"].value_counts(normalize=True).to_dict()
            return {r: float(counts.get(r, 0.0) * 100.0) for r in ["Low", "Moderate", "High", "Extreme"]}

        real_regimes = get_regime_dist(df_real_train)
        synth_regimes = get_regime_dist(df_synthetic)

        # 7. Summary Decisions
        w1_pass = (mean_w1 <= 0.15)
        corr_pass = (frob_dist <= 0.20)
        acf_pass = (mean_acf_error <= 0.08)
        extreme_pass = (extreme_res["coherence_rate"] >= 0.95)

        return {
            "physical_pass": physical_pass,
            "nan_count": nan_count,
            "mean_normalized_w1": mean_w1,
            "mean_ks_stat": mean_ks,
            "w1_pass": w1_pass,
            "per_variable_w1": w1_scores,
            "correlation_frobenius_distance": frob_dist,
            "correlation_pass": corr_pass,
            "mean_acf_error": mean_acf_error,
            "acf_pass": acf_pass,
            "extreme_coherence_rate": extreme_res["coherence_rate"],
            "extreme_pass": extreme_pass,
            "extreme_details": extreme_res,
            "real_regimes_pct": real_regimes,
            "synth_regimes_pct": synth_regimes,
        }
