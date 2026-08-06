import numpy as np
import pandas as pd
from ml.src.features.utils import ensure_chronological


class TimeFeatureExtractor:
    """Generates temporal, calendar, festival proximity, and seasonal features."""

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

        # Festival dates mapping (Diwali dates for 2023 & 2024)
        self.diwali_dates = [pd.Timestamp("2023-11-12"), pd.Timestamp("2024-10-31")]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts time, calendar, seasonal, and festival proximity features."""
        df = ensure_chronological(df, self.date_col)
        dt = pd.to_datetime(df[self.date_col])

        # Cyclical & temporal features
        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["day_of_year"] = dt.dt.dayofyear
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)

        # Seasonal indicators (Delhi climate classification)
        # Winter: Nov-Feb (11, 12, 1, 2)
        # Summer: Mar-Jun (3, 4, 5, 6)
        # Monsoon: Jul-Sep (7, 8, 9)
        # Post-Monsoon: Oct (10)
        df["is_winter"] = df["month"].apply(lambda m: 1 if m in [11, 12, 1, 2] else 0)
        df["is_summer"] = df["month"].apply(lambda m: 1 if m in [3, 4, 5, 6] else 0)
        df["is_monsoon"] = df["month"].apply(lambda m: 1 if m in [7, 8, 9] else 0)
        df["is_post_monsoon"] = df["month"].apply(lambda m: 1 if m == 10 else 0)

        # Stubble burning season: October 15 - November 30
        df["is_stubble_season"] = dt.apply(
            lambda d: 1 if ((d.month == 10 and d.day >= 15) or (d.month == 11)) else 0
        )

        # Diwali Proximity Features (Days until and days since Diwali)
        days_until_list = []
        days_since_list = []
        festival_window_list = []

        for d in dt:
            # Find closest Diwali in future or past
            until_diffs = [(diwali - d).days for diwali in self.diwali_dates if (diwali - d).days >= 0]
            since_diffs = [(d - diwali).days for diwali in self.diwali_dates if (d - diwali).days >= 0]

            days_until = min(until_diffs) if until_diffs else 365
            days_since = min(since_diffs) if since_diffs else 365

            days_until_list.append(days_until)
            days_since_list.append(days_since)

            # Festival window: Diwali day ± 3 days
            in_window = 1 if (days_until <= 3 or days_since <= 3) else 0
            festival_window_list.append(in_window)

        df["days_until_diwali"] = days_until_list
        df["days_since_diwali"] = days_since_list
        df["festival_window"] = festival_window_list

        return df
