import yaml
import pandas as pd
from pathlib import Path
from ml.src.utils.logger import setup_logger

logger = setup_logger("CalendarIngestion")


class CalendarIngestor:
    """Generates calendar, holiday, festival, and stubble burning season flags for Delhi."""

    def __init__(self, config_path: str = "ml/configs/ingestion_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.start_date = self.config["data_range"]["start_date"]
        self.end_date = self.config["data_range"]["end_date"]

        self.output_dir = Path(self.config["paths"]["raw_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "calendar_raw.csv"

        # Indian National Public Holidays (2023 & 2024)
        self.holidays = {
            # 2023
            "2023-01-26": "Republic Day",
            "2023-08-15": "Independence Day",
            "2023-10-02": "Gandhi Jayanti",
            "2023-12-25": "Christmas",
            "2023-01-01": "New Year",
            # 2024
            "2024-01-26": "Republic Day",
            "2024-08-15": "Independence Day",
            "2024-10-02": "Gandhi Jayanti",
            "2024-12-25": "Christmas",
            "2024-01-01": "New Year"
        }

        # Major Festival Days (Firework / traffic spikes)
        self.festivals = {
            # 2023
            "2023-03-07": "Holi",
            "2023-03-08": "Holi",
            "2023-10-24": "Dussehra",
            "2023-11-01": "Karwa Chauth",
            "2023-11-12": "Diwali",
            "2023-11-13": "Diwali",
            "2023-11-19": "Chhath Puja",
            # 2024
            "2024-03-24": "Holi",
            "2024-03-25": "Holi",
            "2024-10-12": "Dussehra",
            "2024-10-20": "Karwa Chauth",
            "2024-10-31": "Diwali",
            "2024-11-01": "Diwali",
            "2024-11-07": "Chhath Puja"
        }

    def generate_calendar(self) -> pd.DataFrame:
        """Generates daily calendar feature flags."""
        logger.info(f"Generating calendar features ({self.start_date} to {self.end_date})...")

        dates = pd.date_range(start=self.start_date, end=self.end_date)
        records = []

        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            month = d.month
            day = d.day
            day_of_week = d.dayofweek  # 0=Mon, 6=Sun

            is_weekend = 1 if day_of_week in [5, 6] else 0
            is_holiday = 1 if d_str in self.holidays else 0
            is_festival = 1 if d_str in self.festivals else 0

            # Stubble burning season: October 15 - November 30
            is_stubble_season = 1 if ((month == 10 and day >= 15) or (month == 11)) else 0

            records.append({
                "date": d_str,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_festival": is_festival,
                "is_stubble_season": is_stubble_season
            })

        df = pd.DataFrame(records)
        logger.info(f"Successfully generated calendar dataset with {len(df)} days.")
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates calendar schema and values."""
        assert len(df) > 0, "Calendar dataset is empty!"
        assert not df["date"].duplicated().any(), "Duplicate dates in calendar!"
        assert set(df["is_weekend"].unique()).issubset({0, 1}), "is_weekend must be binary!"
        assert set(df["is_holiday"].unique()).issubset({0, 1}), "is_holiday must be binary!"
        assert set(df["is_festival"].unique()).issubset({0, 1}), "is_festival must be binary!"
        assert set(df["is_stubble_season"].unique()).issubset({0, 1}), "is_stubble_season must be binary!"

        logger.info("Calendar validation successful.")
        return df

    def run(self) -> Path:
        """Executes calendar generation workflow and saves CSV."""
        df = self.generate_calendar()
        df_valid = self.validate(df)
        df_valid.to_csv(self.output_file, index=False)
        logger.info(f"Calendar raw data saved to: {self.output_file}")
        return self.output_file


if __name__ == "__main__":
    ingestor = CalendarIngestor()
    ingestor.run()
