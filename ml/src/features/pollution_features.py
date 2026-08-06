import numpy as np
import pandas as pd
from ml.src.features.utils import ensure_chronological


class PollutionFeatureExtractor:
    """Engineers ambient air quality ratios, trends, rolling volatility, and anomaly metrics."""

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms pollutant columns into atmospheric indicator features."""
        df = ensure_chronological(df, self.date_col)

        pm25 = df["pm25"]
        pm10 = df["pm10"]
        no2 = df["no2"]
        so2 = df["so2"]
        co = df["co"]

        # 1. Chemical Ratios
        df["pm25_pm10_ratio"] = (pm25 / (pm10 + 1e-5)).clip(0.0, 1.0).round(4)
        df["no2_so2_ratio"] = (no2 / (so2 + 1e-5)).round(3)

        # 2. Normalized CO (Min-Max Scaling)
        co_min = co.min()
        co_max = co.max()
        df["co_normalized"] = ((co - co_min) / (co_max - co_min + 1e-5)).round(4)

        # 3. Daily Pollutant Trend (24-hour PM2.5 rate of change)
        df["daily_pollutant_trend"] = pm25.diff(1).fillna(0.0).round(2)

        # 4. Rolling Metrics & Volatility (Shifted by 1 to prevent target leakage)
        pm25_shifted = pm25.shift(1).fillna(pm25.iloc[0])
        df["pollutant_rolling_avg"] = pm25_shifted.rolling(window=7, min_periods=1).mean().round(2)
        df["pollutant_volatility"] = pm25_shifted.rolling(window=7, min_periods=1).std().fillna(0.0).round(2)

        # 5. Anomaly Scores & Z-score
        pm25_ma_30d = pm25_shifted.rolling(window=30, min_periods=1).mean()
        pm25_std_30d = pm25_shifted.rolling(window=30, min_periods=1).std().fillna(1.0)

        df["pollutant_zscore"] = ((pm25 - pm25_ma_30d) / (pm25_std_30d + 1e-5)).round(3)
        df["pollutant_anomaly_score"] = df["pollutant_zscore"].abs().round(3)

        return df
