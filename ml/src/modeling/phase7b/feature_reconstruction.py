"""
AtmosIQ Phase 7B: Mathematical Feature Reconstruction Engine.
"""

from typing import List
import pandas as pd
import numpy as np


class FeatureReconstructorPhase7B:
    """
    Reconstructs all 35 prediction-safe features mathematically from continuous synthetic trajectories.
    Eliminates independent sampling contradictions.
    """

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def reconstruct_trajectory_features(self, df_trajectory: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw continuous trajectory and computes all 35 prediction-safe features.
        """
        df = df_trajectory.copy().sort_values("step_idx").reset_index(drop=True)

        # 1. PM2.5 persistence lags & rolling features
        df["pm25_lag_1d"] = df["pm25"].shift(1).fillna(df["pm25"])
        df["pm25_lag_2d"] = df["pm25"].shift(2).fillna(df["pm25_lag_1d"])
        df["pm25_lag_3d"] = df["pm25"].shift(3).fillna(df["pm25_lag_2d"])
        df["pm25_lag_7d"] = df["pm25"].shift(7).fillna(df["pm25_lag_3d"])

        df["pm25_roll_mean_3d"] = df["pm25"].rolling(window=3, min_periods=1).mean()
        df["pm25_roll_mean_7d"] = df["pm25"].rolling(window=7, min_periods=1).mean()
        df["pm25_roll_mean_14d"] = df["pm25"].rolling(window=14, min_periods=1).mean()
        df["pm25_roll_std_7d"] = df["pm25"].rolling(window=7, min_periods=1).std().fillna(0.0)
        df["pm25_roll_max_7d"] = df["pm25"].rolling(window=7, min_periods=1).max()
        df["pm25_roll_min_7d"] = df["pm25"].rolling(window=7, min_periods=1).min()

        # 2. Temperature lags & rolling features
        df["temperature_c_lag_1d"] = df["temperature_c"].shift(1).fillna(df["temperature_c"])
        df["temperature_c_roll_mean_3d"] = df["temperature_c"].rolling(window=3, min_periods=1).mean()
        df["temperature_c_roll_min_3d"] = df["temperature_c"].rolling(window=3, min_periods=1).min()

        # 3. Humidity lags & rolling features
        df["humidity_pct_lag_1d"] = df["humidity_pct"].shift(1).fillna(df["humidity_pct"])
        df["humidity_pct_roll_mean_3d"] = df["humidity_pct"].rolling(window=3, min_periods=1).mean()
        df["humidity_pct_roll_max_7d"] = df["humidity_pct"].rolling(window=7, min_periods=1).max()

        # 4. Wind speed lags & rolling features
        df["wind_speed_kmh_lag_1d"] = df["wind_speed_kmh"].shift(1).fillna(df["wind_speed_kmh"])
        df["wind_speed_kmh_roll_mean_3d"] = df["wind_speed_kmh"].rolling(window=3, min_periods=1).mean()

        # 5. Fire count lags & rolling features
        df["fire_hotspot_count_lag_1d"] = df["fire_hotspot_count_1d"].shift(1).fillna(df["fire_hotspot_count_1d"])
        df["fire_hotspot_count_roll_mean_3d"] = df["fire_hotspot_count_1d"].rolling(window=3, min_periods=1).mean()
        df["fire_hotspot_count_roll_mean_7d"] = df["fire_hotspot_count_1d"].rolling(window=7, min_periods=1).mean()

        # 6. Rainfall features
        df["rainfall_3d"] = df["rainfall_1d"].rolling(window=3, min_periods=1).sum()
        # Washout index: log(1 + rainfall_3d) * (humidity / 100)
        df["washout_index_3d"] = np.log1p(df["rainfall_3d"]) * (df["humidity_pct"] / 50.0)

        # 7. Boundary layer features
        df["pblh_roll_mean_3d"] = df["pblh_1d"].rolling(window=3, min_periods=1).mean()

        # Ensure all 35 registry features are present
        for feat in self.feature_registry:
            if feat not in df.columns:
                raise KeyError(f"Reconstructed trajectory missing registry feature: {feat}")

        return df
