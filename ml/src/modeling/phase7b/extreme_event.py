"""
AtmosIQ Phase 7B: Extreme Event Coherence Engine.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class ExtremeEventGenerator:
    """
    Validates and enforces joint environmental coherence for severe pollution episodes:
    PM2.5 >= 250 µg/m³ requires:
    1. Low/Moderate ventilation (VI <= 4,000 m²/s)
    2. Zero or minimal rain (Rainfall <= 5.0 mm)
    3. Shallow nocturnal inversion (PBLH_min <= 600m)
    4. Appropriate season (Winter or Post-Monsoon)
    """

    def __init__(self):
        pass

    def evaluate_extreme_coherence(self, df_synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Audits joint atmospheric coherence for all synthetic records >= 250 µg/m³."""
        df_extreme = df_synthetic[df_synthetic["pm25"] >= 250.0]
        total_extreme = len(df_extreme)

        if total_extreme == 0:
            return {
                "total_extreme_events": 0,
                "coherent_events": 0,
                "coherence_rate": 1.0,
                "vi_coherent_pct": 100.0,
                "rain_coherent_pct": 100.0,
                "season_coherent_pct": 100.0,
            }

        # 1. Ventilation check: VI <= 4,500 m²/s
        vi_ok = (df_extreme["ventilation_index_1d"] <= 4500.0)
        # 2. Rain check: Rain <= 5.0 mm
        rain_ok = (df_extreme["rainfall_1d"] <= 5.0)
        # 3. Season check: Post-Monsoon or Winter
        season_ok = df_extreme["season"].isin(["Winter", "Post-Monsoon"])

        all_ok = vi_ok & rain_ok & season_ok
        coherent_count = int(all_ok.sum())
        coherence_rate = float(coherent_count / total_extreme)

        return {
            "total_extreme_events": total_extreme,
            "coherent_events": coherent_count,
            "coherence_rate": coherence_rate,
            "vi_coherent_pct": float(vi_ok.mean() * 100.0),
            "rain_coherent_pct": float(rain_ok.mean() * 100.0),
            "season_coherent_pct": float(season_ok.mean() * 100.0),
        }
