import time
import yaml
import requests
import pandas as pd
from pathlib import Path
from ml.src.utils.logger import setup_logger

logger = setup_logger("OpenMeteoIngestion")


class OpenMeteoIngestor:
    """Ingests historical meteorology data for Delhi from Open-Meteo API."""

    def __init__(self, config_path: str = "ml/configs/ingestion_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config["sources"]["open_meteo"]["base_url"]
        self.lat = self.config["location"]["latitude"]
        self.lon = self.config["location"]["longitude"]
        self.start_date = self.config["data_range"]["start_date"]
        self.end_date = self.config["data_range"]["end_date"]
        self.timezone = self.config["project"]["timezone"]
        self.retry_attempts = self.config["sources"]["open_meteo"].get("retry_attempts", 3)
        self.retry_delay = self.config["sources"]["open_meteo"].get("retry_delay_seconds", 5)

        self.output_dir = Path(self.config["paths"]["raw_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "open_meteo_raw.csv"

    def fetch_data(self) -> pd.DataFrame:
        """Downloads historical weather data with retry logic."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "daily": ",".join(self.config["sources"]["open_meteo"]["daily_variables"]),
            "timezone": self.timezone
        }

        logger.info(f"Downloading Open-Meteo weather data for Delhi ({self.start_date} to {self.end_date})...")

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    daily_data = data.get("daily", {})
                    df = pd.DataFrame(daily_data)
                    logger.info(f"Successfully downloaded {len(df)} daily weather records from Open-Meteo.")
                    return df
                else:
                    logger.warning(f"Attempt {attempt}: API responded with HTTP status {response.status_code}.")
            except Exception as e:
                logger.warning(f"Attempt {attempt}: Network error during request - {str(e)}")

            if attempt < self.retry_attempts:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to fetch Open-Meteo weather data after {self.retry_attempts} attempts.")

    def process_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and validates meteorology schema."""
        rename_map = {
            "time": "date",
            "temperature_2m_mean": "temperature_c",
            "relative_humidity_2m_mean": "humidity_pct",
            "wind_speed_10m_max": "wind_speed_kmh",
            "wind_direction_10m_dominant": "wind_direction_deg",
            "surface_pressure_mean": "pressure_hpa",
            "precipitation_sum": "precipitation_mm"
        }
        df = df.rename(columns=rename_map)

        # Standardize date format
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Range and null sanity checks
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Null values detected in Open-Meteo data:\n{null_counts[null_counts > 0]}")
            df = df.ffill().bfill()

        # Sanity validation
        assert (df["temperature_c"] >= -10).all() and (df["temperature_c"] <= 55).all(), "Temperature out of physical bounds!"
        assert (df["humidity_pct"] >= 0).all() and (df["humidity_pct"] <= 100).all(), "Humidity out of physical bounds!"
        assert (df["pressure_hpa"] >= 900).all() and (df["pressure_hpa"] <= 1100).all(), "Pressure out of physical bounds!"

        logger.info(f"Validation successful. Final schema: {list(df.columns)}")
        return df

    def run(self) -> Path:
        """Executes ingestion workflow and saves raw CSV."""
        df = self.fetch_data()
        df_clean = self.process_and_validate(df)
        df_clean.to_csv(self.output_file, index=False)
        logger.info(f"Open-Meteo raw data saved to: {self.output_file}")
        return self.output_file


if __name__ == "__main__":
    ingestor = OpenMeteoIngestor()
    ingestor.run()
