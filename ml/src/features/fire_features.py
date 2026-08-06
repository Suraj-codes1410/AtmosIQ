import numpy as np
import pandas as pd
from ml.src.features.utils import ensure_chronological, deg_to_rad
from ml.src.features.feature_config import feature_config


class FireFeatureExtractor:
    """Engineers agricultural residue burning and satellite fire hotspot features."""

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col
        self.nw_wind_deg = feature_config.STUBBLE_BURNING_WIND_DIR_DEG

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw satellite fire metrics into process features."""
        df = ensure_chronological(df, self.date_col)

        fc = df["fire_hotspot_count"]
        mb = df["mean_fire_brightness"]
        hc = df["high_confidence_fire_count"]

        # 1. Rolling Sums & Moving Averages (Shifted by 1 to prevent leakage)
        fc_shifted = fc.shift(1).fillna(0.0)
        df["fire_hotspot_sum_7d"] = fc_shifted.rolling(window=7, min_periods=1).sum()
        df["fire_hotspot_sum_14d"] = fc_shifted.rolling(window=14, min_periods=1).sum()

        df["fire_hotspot_ma_3d"] = fc_shifted.rolling(window=3, min_periods=1).mean()
        df["fire_hotspot_ma_7d"] = fc_shifted.rolling(window=7, min_periods=1).mean()

        # 2. Brightness Trend (3-day vs 7-day mean brightness)
        mb_shifted = mb.shift(1).fillna(300.0)
        mb_ma_3d = mb_shifted.rolling(window=3, min_periods=1).mean()
        mb_ma_7d = mb_shifted.rolling(window=7, min_periods=1).mean()
        df["brightness_trend"] = (mb_ma_3d - mb_ma_7d).round(3)

        # 3. High-Confidence Fire Ratio
        df["high_confidence_fire_ratio"] = (hc / (fc + 1e-5)).clip(0.0, 1.0).round(3)

        # 4. Fire Dynamics: Acceleration & Momentum
        fc_diff1 = fc.diff(1).fillna(0.0)
        fc_diff2 = fc.diff(2).fillna(0.0)
        df["fire_acceleration"] = (fc_diff1 - fc_diff2).round(2)
        df["fire_momentum"] = (fc - df["fire_hotspot_ma_7d"]).round(2)

        # 5. Fire Density & Anomaly Score
        # Regional area approx 150,000 km^2
        df["fire_density"] = (fc / 15.0).round(3)  # Hotspots per 10,000 km^2

        fc_ma_30d = fc_shifted.rolling(window=30, min_periods=1).mean()
        fc_std_30d = fc_shifted.rolling(window=30, min_periods=1).std().fillna(1.0)
        df["fire_anomaly"] = ((fc - fc_ma_30d) / (fc_std_30d + 1e-5)).round(3)

        # 6. Wind-Weighted Hotspot Transport Score
        # Projects wind direction relative to NW stubble corridor (~315 degrees towards Delhi)
        wind_dir_rad = deg_to_rad(df["wind_direction_deg"])
        nw_rad = deg_to_rad(pd.Series(self.nw_wind_deg, index=df.index))

        # Direction alignment: cos(wind_dir - 315 deg)
        alignment = np.maximum(0.0, np.cos(wind_dir_rad - nw_rad))
        df["wind_weighted_hotspot_transport_score"] = (
            fc * df["wind_speed_kmh"] * alignment
        ).round(2)

        # 7. Distance-Weighted Hotspot Score (Centroid Approximation)
        # Stubble burning region centroid (Punjab/Haryana ~250km NW of Delhi)
        approx_distance_km = 250.0
        df["distance_weighted_hotspot_score"] = (fc / (approx_distance_km ** 2)).round(5)

        return df
