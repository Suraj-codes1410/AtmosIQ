import pandas as pd
from ml.src.features.utils import ensure_chronological


class InteractionFeatureExtractor:
    """Engineers non-linear environmental domain interaction features."""

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms primary features into non-linear domain interaction terms."""
        df = ensure_chronological(df, self.date_col)

        fc = df["fire_hotspot_count"]
        ws = df["wind_speed_kmh"]
        temp = df["temperature_c"]
        rh = df["humidity_pct"]
        rain = df["precipitation_mm"]
        press = df["pressure_hpa"]
        fest = df["is_festival"] if "is_festival" in df.columns else pd.Series(0, index=df.index)
        wknd = df["is_weekend"] if "is_weekend" in df.columns else pd.Series(0, index=df.index)
        traffic = df["traffic_activity_proxy"] if "traffic_activity_proxy" in df.columns else pd.Series(1.0, index=df.index)
        month = df["month"] if "month" in df.columns else pd.Series(1, index=df.index)
        transport = df["wind_weighted_hotspot_transport_score"] if "wind_weighted_hotspot_transport_score" in df.columns else fc * ws

        # 1. Fire x Meteorology Interaction Terms
        df["fire_count_x_wind_speed"] = (fc * ws).round(2)
        df["fire_count_x_wind_dir_nw"] = (fc * transport).round(2)

        # 2. Meteorological Process Interaction Terms
        df["temp_x_humidity"] = (temp * rh).round(2)
        df["temp_x_rainfall"] = (temp * rain).round(2)
        df["pressure_x_wind"] = (press * ws).round(2)

        # 3. Socio-Anthropogenic & Seasonal Interaction Terms
        df["festival_x_fire_count"] = (fest * fc).round(2)
        df["weekend_x_traffic_proxy"] = (wknd * traffic).round(2)
        df["month_x_fire_count"] = (month * fc).round(2)

        return df
