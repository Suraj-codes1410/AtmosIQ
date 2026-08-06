import yaml
import pandas as pd
from pathlib import Path
from ml.src.utils.logger import setup_logger
from ml.src.preprocessing.data_validation import DataValidator

logger = setup_logger("MergePipeline")


class MasterMergePipeline:
    """Merges OpenAQ, NASA FIRMS, Open-Meteo, and Calendar raw datasets into processed/master_dataset.csv."""

    def __init__(self, config_path: str = "ml/configs/ingestion_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path(self.config["paths"]["raw_dir"])
        self.processed_dir = Path(self.config["paths"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = Path(self.config["paths"]["master_output"])

        self.openaq_file = self.raw_dir / "openaq_delhi_raw.csv"
        self.firms_file = self.raw_dir / "nasa_firms_raw.csv"
        self.meteo_file = self.raw_dir / "open_meteo_raw.csv"
        self.calendar_file = self.raw_dir / "calendar_raw.csv"

        self.validator = DataValidator(target_tz=self.config["project"]["timezone"])

    def load_raw_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads raw CSV files from raw_dir."""
        logger.info("Loading raw datasets from ml/data/raw/...")

        assert self.openaq_file.exists(), f"Missing file: {self.openaq_file}"
        assert self.firms_file.exists(), f"Missing file: {self.firms_file}"
        assert self.meteo_file.exists(), f"Missing file: {self.meteo_file}"
        assert self.calendar_file.exists(), f"Missing file: {self.calendar_file}"

        df_openaq = pd.read_csv(self.openaq_file)
        df_firms = pd.read_csv(self.firms_file)
        df_meteo = pd.read_csv(self.meteo_file)
        df_calendar = pd.read_csv(self.calendar_file)

        logger.info(f"Loaded OpenAQ: {df_openaq.shape}")
        logger.info(f"Loaded NASA FIRMS: {df_firms.shape}")
        logger.info(f"Loaded Open-Meteo: {df_meteo.shape}")
        logger.info(f"Loaded Calendar: {df_calendar.shape}")

        return df_openaq, df_firms, df_meteo, df_calendar

    def merge_datasets(
        self,
        df_openaq: pd.DataFrame,
        df_firms: pd.DataFrame,
        df_meteo: pd.DataFrame,
        df_calendar: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges all 4 datasets on 'date' timestamp."""
        logger.info("Merging datasets on 'date' primary key...")

        # Normalize dates
        for df in [df_openaq, df_firms, df_meteo, df_calendar]:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Sequentially merge
        df_merged = pd.merge(df_calendar, df_openaq, on="date", how="left")
        df_merged = pd.merge(df_merged, df_meteo, on="date", how="left")
        df_merged = pd.merge(df_merged, df_firms, on="date", how="left")

        # Sort by date chronologically
        df_merged = df_merged.sort_values("date").reset_index(drop=True)

        logger.info(f"Merged Dataset Shape: {df_merged.shape}")
        return df_merged

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values and formats data types."""
        logger.info("Cleaning and imputing minor missing values...")

        # Fill fire hotspot NaNs with 0
        df["fire_hotspot_count"] = df["fire_hotspot_count"].fillna(0).astype(int)
        df["high_confidence_fire_count"] = df["high_confidence_fire_count"].fillna(0).astype(int)
        df["mean_fire_brightness"] = df["mean_fire_brightness"].fillna(300.0).round(2)

        # Forward fill and backward fill for continuous environmental series
        continuous_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature_c", "humidity_pct", "wind_speed_kmh", "wind_direction_deg", "pressure_hpa", "precipitation_mm"]
        df[continuous_cols] = df[continuous_cols].ffill().bfill()

        # Re-check for any remaining NaNs
        remaining_nans = df.isnull().sum().sum()
        assert remaining_nans == 0, f"Error: Found {remaining_nans} remaining NaNs after imputation!"

        logger.info("Cleaning and imputation complete. Zero NaNs remaining.")
        return df

    def run(self) -> Path:
        """Executes full merge pipeline and saves master_dataset.csv."""
        logger.info("=== Starting Master Dataset Merge Pipeline ===")

        df_openaq, df_firms, df_meteo, df_calendar = self.load_raw_datasets()
        df_merged = self.merge_datasets(df_openaq, df_firms, df_meteo, df_calendar)
        df_clean = self.clean_and_impute(df_merged)

        # Validate master dataset
        self.validator.validate_master_dataset(df_clean)

        # Save to processed directory
        df_clean.to_csv(self.output_file, index=False)

        logger.info("=== Master Dataset Merge Pipeline Completed Successfully ===")
        logger.info(f"Master Dataset saved to: {self.output_file}")
        logger.info(f"Rows: {len(df_clean)}, Columns: {len(df_clean.columns)}")
        logger.info(f"Features: {list(df_clean.columns)}")

        return self.output_file


if __name__ == "__main__":
    pipeline = MasterMergePipeline()
    pipeline.run()
