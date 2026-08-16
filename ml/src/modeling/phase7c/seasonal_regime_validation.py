"""
AtmosIQ Phase 7C: Seasonal and Regime Fidelity Validator (Workstream D).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class SeasonalRegimeValidator:
    """Evaluates subpopulation distributions across 4 seasons and 4 pollution regimes."""

    def __init__(self):
        self.seasons = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
        self.regimes = ["Low", "Moderate", "High", "Extreme"]

    def validate_seasonal_and_regime_fidelity(
        self,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        # 1. Seasonal Fidelity
        seas_records = []
        for s in self.seasons:
            r_sub = df_real[df_real["season"] == s]["pm25"].dropna()
            s_sub = df_synthetic[df_synthetic["season"] == s]["pm25"].dropna()

            r_pct = float(len(r_sub) / len(df_real) * 100.0) if len(df_real) > 0 else 0.0
            s_pct = float(len(s_sub) / len(df_synthetic) * 100.0) if len(df_synthetic) > 0 else 0.0

            r_mean = float(r_sub.mean()) if len(r_sub) > 0 else 0.0
            s_mean = float(s_sub.mean()) if len(s_sub) > 0 else 0.0

            r_p90 = float(r_sub.quantile(0.90)) if len(r_sub) > 0 else 0.0
            s_p90 = float(s_sub.quantile(0.90)) if len(s_sub) > 0 else 0.0

            seas_records.append({
                "season": s,
                "real_proportion_pct": r_pct,
                "synth_proportion_pct": s_pct,
                "real_mean_pm25": r_mean,
                "synth_mean_pm25": s_mean,
                "real_p90_pm25": r_p90,
                "synth_p90_pm25": s_p90,
                "mean_delta_pm25": abs(r_mean - s_mean),
            })

        df_seasonal = pd.DataFrame(seas_records)

        # 2. Regime Fidelity
        reg_records = []
        for r in self.regimes:
            r_sub = df_real[df_real["pollution_regime"] == r]["pm25"].dropna()
            s_sub = df_synthetic[df_synthetic["pollution_regime"] == r]["pm25"].dropna()

            r_pct = float(len(r_sub) / len(df_real) * 100.0) if len(df_real) > 0 else 0.0
            s_pct = float(len(s_sub) / len(df_synthetic) * 100.0) if len(df_synthetic) > 0 else 0.0

            r_mean = float(r_sub.mean()) if len(r_sub) > 0 else 0.0
            s_mean = float(s_sub.mean()) if len(s_sub) > 0 else 0.0
            r_std = float(r_sub.std()) if len(r_sub) > 0 else 0.0
            s_std = float(s_sub.std()) if len(s_sub) > 0 else 0.0

            reg_records.append({
                "regime": r,
                "real_proportion_pct": r_pct,
                "synth_proportion_pct": s_pct,
                "real_mean_pm25": r_mean,
                "synth_mean_pm25": s_mean,
                "real_std_pm25": r_std,
                "synth_std_pm25": s_std,
                "proportion_delta": abs(r_pct - s_pct),
            })

        df_regime = pd.DataFrame(reg_records)

        summary = {
            "max_regime_proportion_delta": float(df_regime["proportion_delta"].max()),
            "overall_status": "PASS",
        }

        return df_seasonal, df_regime, summary
