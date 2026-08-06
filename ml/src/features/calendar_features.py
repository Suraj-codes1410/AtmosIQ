import pandas as pd
from ml.src.features.time_features import TimeFeatureExtractor


class CalendarFeatureExtractor:
    """Extracts and validates calendar and holiday features."""

    def __init__(self, date_col: str = "date"):
        self.time_extractor = TimeFeatureExtractor(date_col=date_col)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms calendar feature matrix."""
        df = self.time_extractor.transform(df)

        # Traffic proxy: 0 on weekends/holidays, 1 on weekday working days
        if "is_holiday" in df.columns:
            df["traffic_activity_proxy"] = df.apply(
                lambda row: 0.3 if (row["is_weekend"] == 1 or row["is_holiday"] == 1) else 1.0,
                axis=1
            )
        else:
            df["traffic_activity_proxy"] = df["is_weekend"].apply(lambda w: 0.3 if w == 1 else 1.0)

        return df
