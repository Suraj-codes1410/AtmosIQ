import os
from pathlib import Path


class FeatureConfig:
    """Feature Engineering Configuration Parameters."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    INPUT_PATH: Path = BASE_DIR / "ml" / "data" / "processed" / "master_dataset.csv"
    OUTPUT_PATH: Path = BASE_DIR / "ml" / "data" / "processed" / "feature_dataset.csv"

    TARGET_COL: str = "pm25"
    DATE_COL: str = "date"

    # Lags configuration
    LAG_WINDOWS: list[int] = [1, 2, 3, 7, 14]
    LAG_COLUMNS: list[str] = [
        "pm25", "pm10", "no2", "temperature_c", "humidity_pct",
        "wind_speed_kmh", "fire_hotspot_count", "precipitation_mm"
    ]

    # Rolling window configuration
    ROLLING_WINDOWS: list[int] = [3, 7, 14, 30]
    ROLLING_COLUMNS: list[str] = [
        "pm25", "pm10", "temperature_c", "humidity_pct",
        "fire_hotspot_count", "wind_speed_kmh"
    ]
    ROLLING_FUNCS: list[str] = ["mean", "median", "max", "min", "std", "var"]

    # Wind transport corridor parameters (Northwest direction: ~315 degrees)
    STUBBLE_BURNING_WIND_DIR_DEG: float = 315.0


feature_config = FeatureConfig()
