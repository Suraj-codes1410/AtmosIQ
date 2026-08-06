import pandas as pd
from pathlib import Path
from ml.src.preprocessing.data_validation import DataValidator


def test_data_files_exist():
    raw_dir = Path("ml/data/raw")
    processed_dir = Path("ml/data/processed")

    assert (raw_dir / "openaq_delhi_raw.csv").exists()
    assert (raw_dir / "nasa_firms_raw.csv").exists()
    assert (raw_dir / "open_meteo_raw.csv").exists()
    assert (raw_dir / "calendar_raw.csv").exists()
    assert (processed_dir / "master_dataset.csv").exists()


def test_master_dataset_schema():
    df = pd.read_csv("ml/data/processed/master_dataset.csv")

    expected_cols = [
        "date", "day_of_week", "is_weekend", "is_holiday", "is_festival", "is_stubble_season",
        "pm25", "pm10", "no2", "so2", "co", "o3",
        "temperature_c", "humidity_pct", "wind_speed_kmh", "wind_direction_deg", "pressure_hpa", "precipitation_mm",
        "fire_hotspot_count", "mean_fire_brightness", "high_confidence_fire_count"
    ]

    assert len(df) == 731, f"Expected 731 daily rows, got {len(df)}"
    for col in expected_cols:
        assert col in df.columns, f"Missing required master column: {col}"

    assert df.isnull().sum().sum() == 0, "Master dataset contains unexpected NaNs!"


def test_data_validator_checks():
    validator = DataValidator()
    df = pd.read_csv("ml/data/processed/master_dataset.csv")

    assert validator.check_duplicate_timestamps(df, "date") is True
    assert validator.check_unit_consistency(df) is True
