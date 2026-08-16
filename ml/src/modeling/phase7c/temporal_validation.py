"""
AtmosIQ Phase 7C: Temporal Dynamics Validator (Workstream C).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class TemporalDynamicsValidator:
    """Evaluates multi-day autocorrelation, persistence, and regime transition dynamics."""

    def __init__(self):
        pass

    def validate_temporal_dynamics(
        self,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray]:
        def compute_acf(s, max_lag=30):
            s = s - np.mean(s)
            var = np.var(s)
            if var == 0: return np.ones(max_lag)
            return [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in range(1, max_lag + 1)]

        acf_r = compute_acf(df_real["pm25"].dropna().values, 30)
        acf_s = compute_acf(df_synthetic["pm25"].dropna().values, 30)

        acf_records = []
        for lag in range(1, 31):
            r_val = acf_r[lag - 1]
            s_val = acf_s[lag - 1]
            err = abs(r_val - s_val)
            acf_records.append({
                "lag": lag,
                "real_acf": r_val,
                "synth_acf": s_val,
                "absolute_error": err,
                "status": "PASS" if err <= 0.15 else "WARNING",
            })

        df_acf = pd.DataFrame(acf_records)
        mean_acf_err_7 = float(df_acf["absolute_error"].iloc[:7].mean())
        mean_acf_err_30 = float(df_acf["absolute_error"].mean())
        max_acf_err = float(df_acf["absolute_error"].max())

        # Regime dwell times
        def calculate_dwell_times(regimes):
            dwells = []
            cur_r = regimes[0]
            cur_len = 1
            for r in regimes[1:]:
                if r == cur_r:
                    cur_len += 1
                else:
                    dwells.append((cur_r, cur_len))
                    cur_r = r
                    cur_len = 1
            dwells.append((cur_r, cur_len))
            return pd.DataFrame(dwells, columns=["regime", "dwell_days"])

        r_reg = df_real["pollution_regime"].values if "pollution_regime" in df_real.columns else []
        s_reg = df_synthetic["pollution_regime"].values if "pollution_regime" in df_synthetic.columns else []

        dwell_real_mean = float(calculate_dwell_times(r_reg)["dwell_days"].mean()) if len(r_reg) > 0 else 0.0
        dwell_synth_mean = float(calculate_dwell_times(s_reg)["dwell_days"].mean()) if len(s_reg) > 0 else 0.0

        summary = {
            "mean_acf_error_lags_1_7": mean_acf_err_7,
            "mean_acf_error_lags_1_30": mean_acf_err_30,
            "max_acf_error": max_acf_err,
            "real_mean_regime_dwell_days": dwell_real_mean,
            "synth_mean_regime_dwell_days": dwell_synth_mean,
            "overall_status": "PASS" if mean_acf_err_7 <= 0.08 else "WARNING",
        }

        return df_acf, summary, np.array(acf_r), np.array(acf_s)
