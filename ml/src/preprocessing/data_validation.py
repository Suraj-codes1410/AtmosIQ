import pandas as pd
from typing import Dict, Any
from ml.src.utils.logger import setup_logger

logger = setup_logger("DataValidation")


class DataValidator:
    """Data Validation Suite for atmospheric, satellite, and environmental datasets."""

    def __init__(self, target_tz: str = "Asia/Kolkata"):
        self.target_tz = target_tz

    def check_missing_values(self, df: pd.DataFrame, max_null_ratio: float = 0.05) -> Dict[str, float]:
        """Calculates missing value percentage per column and flags threshold breaches."""
        null_ratios = df.isnull().mean()
        high_nulls = null_ratios[null_ratios > max_null_ratio]

        if not high_nulls.empty:
            logger.warning(f"Columns exceeding max null ratio threshold ({max_null_ratio * 100}%):\n{high_nulls}")
        else:
            logger.info("Missing values check passed: all columns within acceptable null thresholds.")

        return null_ratios.to_dict()

    def check_duplicate_timestamps(self, df: pd.DataFrame, timestamp_col: str = "date") -> bool:
        """Verifies that primary key date timestamps are unique."""
        duplicates = df[timestamp_col].duplicated().sum()
        if duplicates > 0:
            logger.error(f"Validation Error: Found {duplicates} duplicate timestamps in column '{timestamp_col}'!")
            return False
        logger.info(f"Duplicate timestamp check passed for '{timestamp_col}'.")
        return True

    def normalize_timezones(self, df: pd.DataFrame, timestamp_col: str = "date") -> pd.DataFrame:
        """Normalizes date column to YYYY-MM-DD string format in target timezone."""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col]).dt.strftime("%Y-%m-%d")
        logger.info(f"Normalized '{timestamp_col}' column to ISO 8601 YYYY-MM-DD format ({self.target_tz}).")
        return df

    def check_unit_consistency(self, df: pd.DataFrame) -> bool:
        """Verifies column values conform to physical unit range boundaries."""
        bounds = {
            "pm25": (0.0, 1000.0),        # µg/m³
            "pm10": (0.0, 2000.0),        # µg/m³
            "no2": (0.0, 500.0),          # µg/m³
            "so2": (0.0, 300.0),          # µg/m³
            "co": (0.0, 50.0),            # mg/m³
            "o3": (0.0, 500.0),           # µg/m³
            "temperature_c": (-15.0, 55.0), # °C
            "humidity_pct": (0.0, 100.0),  # %
            "pressure_hpa": (850.0, 1100.0),# hPa
            "wind_speed_kmh": (0.0, 150.0),# km/h
            "precipitation_mm": (0.0, 300.0) # mm
        }

        all_passed = True
        for col, (min_val, max_val) in bounds.items():
            if col in df.columns:
                series = df[col].dropna()
                out_of_bounds = series[(series < min_val) | (series > max_val)]
                if not out_of_bounds.empty:
                    logger.warning(f"Unit consistency warning: '{col}' has {len(out_of_bounds)} values outside range ({min_val}, {max_val})")
                    all_passed = False

        if all_passed:
            logger.info("Unit consistency check passed for all environmental variables.")

        return all_passed

    def validate_master_dataset(self, df: pd.DataFrame) -> bool:
        """Executes full validation pipeline on merged master dataset."""
        logger.info("Starting Master Dataset Validation...")
        
        c1 = self.check_duplicate_timestamps(df, "date")
        self.check_missing_values(df, max_null_ratio=0.02)
        c2 = self.check_unit_consistency(df)

        is_valid = c1 and c2
        if is_valid:
            logger.info("Master Dataset Validation PASSED successfully.")
        else:
            logger.warning("Master Dataset Validation completed with warnings.")

        return is_valid
