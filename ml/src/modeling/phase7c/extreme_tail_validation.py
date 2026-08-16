"""
AtmosIQ Phase 7C: Extreme Tail Fidelity Validator (Workstream E).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class ExtremeTailValidator:
    """Evaluates upper-tail behavior across thresholds >=100 to >=350 µg/m³ and atmospheric coherence."""

    def __init__(self):
        self.thresholds = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0]

    def validate_extreme_tail(
        self,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        tail_records = []

        for th in self.thresholds:
            r_mask = (df_real["pm25"] >= th)
            s_mask = (df_synthetic["pm25"] >= th)

            r_cnt = int(r_mask.sum())
            s_cnt = int(s_mask.sum())

            r_pct = float(r_cnt / len(df_real) * 100.0) if len(df_real) > 0 else 0.0
            s_pct = float(s_cnt / len(df_synthetic) * 100.0) if len(df_synthetic) > 0 else 0.0

            r_sub = df_real[r_mask]
            s_sub = df_synthetic[s_mask]

            r_mean_vi = float(r_sub["ventilation_index_1d"].mean()) if r_cnt > 0 and "ventilation_index_1d" in r_sub else 0.0
            s_mean_vi = float(s_sub["ventilation_index_1d"].mean()) if s_cnt > 0 and "ventilation_index_1d" in s_sub else 0.0

            s_vi_ok = float((s_sub["ventilation_index_1d"] <= 4500.0).mean() * 100.0) if s_cnt > 0 else 100.0
            s_rain_ok = float((s_sub["rainfall_1d"] <= 5.0).mean() * 100.0) if s_cnt > 0 else 100.0

            tail_records.append({
                "threshold_ug_m3": th,
                "real_count": r_cnt,
                "synth_count": s_cnt,
                "real_pct": r_pct,
                "synth_pct": s_pct,
                "real_mean_vi": r_mean_vi,
                "synth_mean_vi": s_mean_vi,
                "synth_vi_coherent_pct": s_vi_ok,
                "synth_rain_coherent_pct": s_rain_ok,
            })

        df_tail = pd.DataFrame(tail_records)

        # 250+ coherence
        sub_250 = df_synthetic[df_synthetic["pm25"] >= 250.0]
        if len(sub_250) > 0:
            coherent_250 = (sub_250["ventilation_index_1d"] <= 4500.0) & (sub_250["rainfall_1d"] <= 5.0) & sub_250["season"].isin(["Winter", "Post-Monsoon"])
            coherence_rate = float(coherent_250.mean())
        else:
            coherence_rate = 1.0

        summary = {
            "extreme_250_count": int((df_synthetic["pm25"] >= 250.0).sum()),
            "extreme_250_coherence_rate": coherence_rate,
            "overall_status": "PASS" if coherence_rate >= 0.95 else "WARNING",
        }

        return df_tail, summary
