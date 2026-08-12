import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class DatasetV2Config:
    """
    AtmosIQ Dataset v2 Configuration.
    Defines 5-year temporal coverage (2020-01-01 to 2024-12-31, 1827 days),
    canonical units, station metadata, regional bounding boxes, and output paths.
    """

    PROJECT_NAME = "AtmosIQ"
    DATASET_VERSION = "v2"
    TIMEZONE = "Asia/Kolkata"

    # Temporal Coverage: 2020-01-01 to 2024-12-31 (5 calendar years = 1827 days)
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    EXPECTED_DAYS = 1827

    # Spatial Bounds (Delhi NCR & Regional Upwind Transport Regions)
    DELHI_LAT = 28.6139
    DELHI_LON = 77.2090

    # Regional Fire Bounding Boxes
    FIRE_REGIONS = {
        "PUNJAB": {"min_lat": 29.5, "max_lat": 32.5, "min_lon": 74.0, "max_lon": 76.5},
        "HARYANA": {"min_lat": 27.6, "max_lat": 30.9, "min_lon": 74.5, "max_lon": 77.6},
        "RAJASTHAN": {"min_lat": 26.0, "max_lat": 29.5, "min_lon": 72.0, "max_lon": 76.0},
        "DELHI_NCR": {"min_lat": 28.2, "max_lat": 29.0, "min_lon": 76.8, "max_lon": 77.5}
    }

    # Station Metadata (Delhi CPCB monitoring stations)
    STATIONS = [
        {"id": "DEL_001", "name": "ITO, Delhi", "lat": 28.6286, "lon": 77.2410},
        {"id": "DEL_002", "name": "Anand Vihar, Delhi", "lat": 28.6470, "lon": 77.3160},
        {"id": "DEL_003", "name": "RK Puram, Delhi", "lat": 28.5630, "lon": 77.1860},
        {"id": "DEL_004", "name": "Punjabi Bagh, Delhi", "lat": 28.6740, "lon": 77.1310},
        {"id": "DEL_005", "name": "Mandir Marg, Delhi", "lat": 28.6364, "lon": 77.1990}
    ]

    # Canonical Measurement Units
    CANONICAL_UNITS = {
        "pm25": "µg/m³",
        "pm10": "µg/m³",
        "no2": "µg/m³",
        "so2": "µg/m³",
        "co": "mg/m³",
        "o3": "µg/m³",
        "temperature_c": "°C",
        "humidity_pct": "%",
        "wind_speed_kmh": "km/h",
        "wind_direction_deg": "degrees",
        "pressure_hpa": "hPa",
        "precipitation_mm": "mm"
    }

    # Output Directory Paths
    BASE_DIR = ROOT_DIR
    RAW_V2_DIR = BASE_DIR / "ml" / "data" / "raw" / "v2"
    INTERMEDIATE_V2_DIR = BASE_DIR / "ml" / "data" / "intermediate" / "v2"
    PROCESSED_V2_DIR = BASE_DIR / "ml" / "data" / "processed" / "v2"
    MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
    EXPERIMENT_V2_DIR = BASE_DIR / "ml" / "experiments" / "phase3e"
    KAGGLE_DIR = BASE_DIR / "kaggle"

    # Train / Validation / Test Chronological Boundaries for Dataset v2
    TRAIN_START = "2020-01-01"
    TRAIN_END = "2022-12-31"
    TRAIN_ROWS = 1096

    VAL_START = "2023-01-01"
    VAL_END = "2023-12-31"
    VAL_ROWS = 365

    TEST_START = "2024-01-01"
    TEST_END = "2024-12-31"
    TEST_ROWS = 366  # 2024 is a leap year!
