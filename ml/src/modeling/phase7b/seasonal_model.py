"""
AtmosIQ Phase 7B: Seasonal and Calendar Dynamics Model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class SeasonalCalendarModel:
    SEASONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

    def __init__(self, random_state: np.random.RandomState):
        self.rng = random_state
        self.seasonal_met_priors: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit_from_training_data(self, df_train: pd.DataFrame):
        """Fit empirical meteorological mean and std per season strictly from training partition."""
        df = df_train.copy()
        if "season" not in df.columns:
            def classify_season(m):
                if m in [12, 1, 2]: return "Winter"
                if m in [3, 4, 5]: return "Summer"
                if m in [6, 7, 8, 9]: return "Monsoon"
                return "Post-Monsoon"
            df["month"] = pd.to_datetime(df["date"]).dt.month
            df["season"] = df["month"].apply(classify_season)

        variables = [
            "temperature_c", "humidity_pct", "wind_speed_kmh",
            "pblh_1d", "pblh_min_1d", "aod_550_1d", "rainfall_1d"
        ]

        for season in self.SEASONS:
            sub = df[df["season"] == season]
            priors = {}
            for var in variables:
                if var in sub.columns:
                    mean_val = float(sub[var].mean())
                    std_val = max(float(sub[var].std()), 1e-4)
                    priors[var] = (mean_val, std_val)
                else:
                    priors[var] = (0.0, 1.0)
            self.seasonal_met_priors[season] = priors

    def get_seasonal_context(self, season: str, step_idx: int) -> Dict[str, Any]:
        """Generate deterministic calendar flags and baseline priors for a seasonal step."""
        is_stubble = 1 if season == "Post-Monsoon" else 0
        festival = 1 if (season == "Post-Monsoon" and step_idx in [5, 6, 7]) else 0

        priors = self.seasonal_met_priors.get(season, {})
        return {
            "season": season,
            "is_stubble_season": is_stubble,
            "festival_window": festival,
            "priors": priors,
        }
