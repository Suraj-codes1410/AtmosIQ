import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root .env if present
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


class Config:
    """Base configuration for atmosIQ ML subsystem."""
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "atmosIQ")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Directory Paths
    BASE_DIR: Path = BASE_DIR
    ML_DIR: Path = BASE_DIR / "ml"
    DATA_DIR: Path = ML_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
    MODELS_DIR: Path = ML_DIR / "models"
    CONFIGS_DIR: Path = ML_DIR / "configs"

    # Database Settings
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "atmosiq_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    DATABASE_URI: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "delhi_aqi_attribution")

    # Model Hyperparameter Default Placeholders
    TARGET_COLUMN: str = "PM2.5"
    RANDOM_SEED: int = 42


config = Config()
