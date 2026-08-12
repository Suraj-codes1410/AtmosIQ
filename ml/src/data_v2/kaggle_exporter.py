import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("KaggleExporterV2")


class KaggleDatasetExporterV2:
    """
    AtmosIQ Dataset v2 Kaggle Public Release Exporter.
    Exports clean, reproducible public release dataset and documentation under kaggle/.
    """

    def __init__(self, modeling_dir: str = "ml/data/modeling/v2", kaggle_dir: str = "kaggle"):
        self.modeling_dir = Path(modeling_dir)
        self.kaggle_dir = Path(kaggle_dir)
        self.kaggle_dir.mkdir(parents=True, exist_ok=True)

    def export_kaggle_dataset(self):
        """Exports main CSV, data dictionary, README, LICENSE, and methodology."""
        logger.info("Preparing clean public-release Kaggle dataset under kaggle/...")
        frozen_file = self.modeling_dir / "feature_dataset_frozen.csv"
        assert frozen_file.exists(), f"Frozen dataset missing: {frozen_file}"

        df = pd.read_csv(frozen_file)
        main_kaggle_file = self.kaggle_dir / "atmosiq_delhi_pm25.csv"
        df.to_csv(main_kaggle_file, index=False)
        logger.info(f"Main Kaggle CSV exported to: {main_kaggle_file} ({len(df)} rows x {len(df.columns)} columns).")

        # 1. Data Dictionary
        dict_rows = [
            {"feature_name": "date", "source": "Timeline", "unit": "YYYY-MM-DD", "type": "Categorical/Date", "description": "Daily observation timestamp (2020-01-01 to 2024-12-31)"},
            {"feature_name": "pm25", "source": "CPCB / OpenAQ", "unit": "µg/m³", "type": "Continuous", "description": "Daily mean PM2.5 concentration in Delhi NCR (Target variable)"},
            {"feature_name": "pm10", "source": "CPCB / OpenAQ", "unit": "µg/m³", "type": "Continuous", "description": "Daily mean PM10 concentration"},
            {"feature_name": "no2", "source": "CPCB / OpenAQ", "unit": "µg/m³", "type": "Continuous", "description": "Daily mean Nitrogen Dioxide concentration"},
            {"feature_name": "temperature_c", "source": "Open-Meteo ERA5", "unit": "°C", "type": "Continuous", "description": "24-hour mean surface air temperature"},
            {"feature_name": "humidity_pct", "source": "Open-Meteo ERA5", "unit": "%", "type": "Continuous", "description": "24-hour mean relative humidity"},
            {"feature_name": "wind_speed_kmh", "source": "Open-Meteo ERA5", "unit": "km/h", "type": "Continuous", "description": "24-hour maximum wind speed"},
            {"feature_name": "fire_count", "source": "NASA FIRMS", "unit": "count", "type": "Integer", "description": "Regional upwind satellite fire hotspot count (Punjab, Haryana, Rajasthan)"},
            {"feature_name": "is_stubble_season", "source": "Calendar Generator", "unit": "binary", "type": "Integer", "description": "Post-monsoon stubble burning period indicator (Oct 15 - Nov 25)"}
        ]
        dict_df = pd.DataFrame(dict_rows)
        dict_df.to_csv(self.kaggle_dir / "atmosiq_data_dictionary.csv", index=False)

        # 2. README.md
        readme_content = """# AtmosIQ: Delhi NCR Daily PM2.5 Air Quality Dataset (2020–2024)

## Overview
AtmosIQ is an explainable AI and atmospheric intelligence platform for Delhi NCR air quality forecasting.
This dataset contains **1,827 continuous daily observations** spanning 5 complete calendar years (January 1, 2020 to December 31, 2024).

## Data Sources & Provenance
- **Ground Air Quality**: CPCB / OpenAQ Delhi station network
- **Meteorology**: Open-Meteo ERA5 historical reanalysis
- **Satellite Biomass Fires**: NASA FIRMS MODIS/VIIRS active fire hotspots
- **Calendar & Seasonality**: Festival proximity (Diwali) and agricultural stubble burning windows

## Leakage Prevention
For 24-hour forecast models ($t-1 \rightarrow t$), all historical lag and rolling statistics are strictly shifted by $\ge 1$ day.

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0).
"""
        with open(self.kaggle_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 3. LICENSE
        license_content = """Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially.
Under the condition that you give appropriate credit to AtmosIQ.
"""
        with open(self.kaggle_dir / "LICENSE", "w", encoding="utf-8") as f:
            f.write(license_content)

        # 4. Methodology.md
        methodology_content = """# AtmosIQ Dataset v2 Technical Methodology

## 1. Ground Air Quality Aggregation
Station-level PM2.5 readings from Delhi CPCB monitoring stations (ITO, Anand Vihar, RK Puram, Punjabi Bagh, Mandir Marg) were aggregated into daily city-wide regional averages.

## 2. Satellite Fire Exposure Proxies
Regional upwind satellite active fire hotspots across Punjab, Haryana, Rajasthan, and Delhi NCR are processed into daily hotspot sums and wind-weighted transport exposure scores.

## 3. Strict Leakage Protection
Target-derived rolling statistics use $t-1 \dots t-k$ historical windows exclusively.
"""
        with open(self.kaggle_dir / "methodology.md", "w", encoding="utf-8") as f:
            f.write(methodology_content)

        logger.info("Kaggle public release artifacts successfully exported.")


if __name__ == "__main__":
    exporter = KaggleDatasetExporterV2()
    exporter.export_kaggle_dataset()
