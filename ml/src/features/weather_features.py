import numpy as np
import pandas as pd
from ml.src.features.utils import ensure_chronological, deg_to_rad


class WeatherFeatureExtractor:
    """Engineers meteorological, atmospheric stability, and wind vector features."""

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw meteorological columns into process features."""
        df = ensure_chronological(df, self.date_col)

        # 1. Wind Vector Components (U = East-West, V = North-South)
        # Meteorological wind direction: direction wind is coming FROM.
        wind_rad = deg_to_rad(df["wind_direction_deg"])
        df["wind_x"] = -df["wind_speed_kmh"] * np.sin(wind_rad)
        df["wind_y"] = -df["wind_speed_kmh"] * np.cos(wind_rad)

        # 2. Temperature-Humidity Index (THI / Thermal Comfort Proxy)
        # Formula: T - (0.55 - 0.55 * RH/100) * (T - 14.5)
        t = df["temperature_c"]
        rh = df["humidity_pct"]
        df["temperature_humidity_index"] = t - (0.55 - 0.55 * (rh / 100.0)) * (t - 14.5)

        # 3. Rainfall Indicators & Consecutive Weather Sequences
        df["is_raining"] = (df["precipitation_mm"] > 0.1).astype(int)

        # Consecutive rain days and dry day count
        rain_mask = df["is_raining"] == 1
        dry_mask = df["is_raining"] == 0

        # Calculate consecutive sequences
        consecutive_rain = []
        consecutive_dry = []

        c_rain = 0
        c_dry = 0
        for is_r in df["is_raining"]:
            if is_r == 1:
                c_rain += 1
                c_dry = 0
            else:
                c_dry += 1
                c_rain = 0
            consecutive_rain.append(c_rain)
            consecutive_dry.append(c_dry)

        df["consecutive_rain_days"] = consecutive_rain
        df["dry_day_count"] = consecutive_dry

        # 4. Daily Meteorological Differences (24-hour rate of change)
        df["temperature_change"] = df["temperature_c"].diff().fillna(0.0)
        df["humidity_change"] = df["humidity_pct"].diff().fillna(0.0)
        df["pressure_change"] = df["pressure_hpa"].diff().fillna(0.0)
        df["wind_speed_change"] = df["wind_speed_kmh"].diff().fillna(0.0)

        # Angular wind direction change (-180 to +180 degrees)
        angle_diff = (df["wind_direction_deg"] - df["wind_direction_deg"].shift(1) + 180) % 360 - 180
        df["wind_direction_change"] = angle_diff.fillna(0.0)

        return df
