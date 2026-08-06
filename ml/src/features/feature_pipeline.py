import yaml
import pandas as pd
from pathlib import Path
from ml.src.utils.logger import setup_logger
from ml.src.features.feature_config import feature_config
from ml.src.features.utils import (
    ensure_chronological,
    create_lags,
    create_rolling_stats,
    clean_inf_and_nans
)
from ml.src.features.calendar_features import CalendarFeatureExtractor
from ml.src.features.weather_features import WeatherFeatureExtractor
from ml.src.features.fire_features import FireFeatureExtractor
from ml.src.features.pollution_features import PollutionFeatureExtractor
from ml.src.features.interaction_features import InteractionFeatureExtractor

logger = setup_logger("FeatureEngineeringPipeline")


class FeatureEngineeringPipeline:
    """Production-grade modular feature engineering pipeline for atmosIQ."""

    def __init__(self, input_path: str = None, output_path: str = None):
        self.input_path = Path(input_path) if input_path else feature_config.INPUT_PATH
        self.output_path = Path(output_path) if output_path else feature_config.OUTPUT_PATH

        self.calendar_extractor = CalendarFeatureExtractor()
        self.weather_extractor = WeatherFeatureExtractor()
        self.fire_extractor = FireFeatureExtractor()
        self.pollution_extractor = PollutionFeatureExtractor()
        self.interaction_extractor = InteractionFeatureExtractor()

    def load_data(self) -> pd.DataFrame:
        """Loads processed master dataset."""
        logger.info(f"Loading master dataset from: {self.input_path}")
        assert self.input_path.exists(), f"Input file not found: {self.input_path}"
        df = pd.read_csv(self.input_path)
        logger.info(f"Master dataset loaded. Initial shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies modular feature engineering transformations sequentially."""
        logger.info("Starting modular feature engineering transformations...")

        df = ensure_chronological(df)

        # 1. Temporal & Calendar Features
        logger.info("Applying Calendar & Temporal transformations...")
        df = self.calendar_extractor.transform(df)

        # 2. Weather & Wind Vector Features
        logger.info("Applying Weather & Wind Vector transformations...")
        df = self.weather_extractor.transform(df)

        # 3. Satellite Fire Hotspot Features
        logger.info("Applying Satellite Fire Hotspot transformations...")
        df = self.fire_extractor.transform(df)

        # 4. Air Quality Pollution Features
        logger.info("Applying Air Quality Pollution transformations...")
        df = self.pollution_extractor.transform(df)

        # 5. Non-linear Interaction Features
        logger.info("Applying Interaction Feature transformations...")
        df = self.interaction_extractor.transform(df)

        # 6. Historical Lag Features
        logger.info(f"Generating Historical Lags ({feature_config.LAG_WINDOWS} days)...")
        df = create_lags(
            df=df,
            cols=feature_config.LAG_COLUMNS,
            lags=feature_config.LAG_WINDOWS
        )

        # 7. Rolling Window Statistics
        logger.info(f"Generating Rolling Window Statistics ({feature_config.ROLLING_WINDOWS} days)...")
        df = create_rolling_stats(
            df=df,
            cols=feature_config.ROLLING_COLUMNS,
            windows=feature_config.ROLLING_WINDOWS,
            funcs=feature_config.ROLLING_FUNCS
        )

        # 8. Clean infinite values and initial boundary NaNs
        logger.info("Cleaning infs and initial boundary NaNs...")
        df = clean_inf_and_nans(df)

        logger.info(f"Transformations complete. Final Feature Matrix Shape: {df.shape}")
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Validates feature dataset schema and integrity."""
        logger.info("Validating feature dataset schema and integrity...")

        assert len(df) == 731, f"Expected 731 rows, got {len(df)}"
        assert not df["date"].duplicated().any(), "Duplicate dates detected!"
        assert df.isnull().sum().sum() == 0, "Feature dataset contains unexpected NaNs!"
        assert "pm25" in df.columns, "Target variable 'pm25' is missing!"

        logger.info("Feature Dataset Validation PASSED successfully.")
        return True

    def export(self, df: pd.DataFrame) -> Path:
        """Saves feature dataset to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Feature dataset successfully saved to: {self.output_path}")
        return self.output_path

    def run(self) -> Path:
        """Executes full feature engineering pipeline end-to-end."""
        logger.info("=== Starting Feature Engineering Pipeline ===")
        df_raw = self.load_data()
        df_feat = self.transform(df_raw)
        self.validate(df_feat)
        out_path = self.export(df_feat)
        logger.info("=== Feature Engineering Pipeline Completed Successfully ===")
        return out_path


if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    pipeline.run()
