import sys
from pathlib import Path
from typing import Optional
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.cache import CachePhase4E
from ml.src.modeling.phase4e.response_schema import PredictionResponse


class PredictionServicePhase4E:
    """
    AtmosIQ Phase 4E Prediction Service.
    Serves model predictions, baseline persistence metrics, and observed pollution errors.
    """

    def __init__(self, data_loader: DataLoaderPhase4E = None, cache: CachePhase4E = None):
        self.loader = data_loader or DataLoaderPhase4E()
        self.cache = cache or CachePhase4E(self.loader)

    def predict_date(self, date_str: str) -> PredictionResponse:
        """Serves prediction for a single YYYY-MM-DD date."""
        if date_str not in self.cache.date_to_index:
            raise ValueError(f"DATE_NOT_FOUND: Date '{date_str}' not found in Dataset v2 (2020-01-01 to 2024-12-31).")

        idx = self.cache.date_to_index[date_str]
        row = self.loader.df_v2.iloc[idx]

        # Model feature row
        x_row = self.loader.X_v2.iloc[[idx]]
        pred_val = float(self.loader.model.predict(x_row)[0])

        obs_val = float(row["pm25"]) if "pm25" in row and pd.notnull(row["pm25"]) else None
        pers_val = float(row["pm25_lag_1d"]) if "pm25_lag_1d" in row and pd.notnull(row["pm25_lag_1d"]) else None
        error_val = pred_val - obs_val if obs_val is not None else None

        # Categorize pollution level
        if obs_val is not None and obs_val >= 306.81:
            category = "Extreme Episode"
        elif pred_val >= 300:
            category = "Severe+"
        elif pred_val >= 200:
            category = "Very Poor"
        elif pred_val >= 100:
            category = "Poor"
        elif pred_val >= 60:
            category = "Moderate"
        else:
            category = "Satisfactory"

        return PredictionResponse(
            date=date_str,
            observed_pm25=obs_val,
            predicted_pm25=round(pred_val, 2),
            persistence_baseline=round(pers_val, 2) if pers_val is not None else None,
            prediction_error=round(error_val, 2) if error_val is not None else None,
            model_version="phase3g_rf_v1",
            dataset_version="v2",
            pollution_category=category
        )


if __name__ == "__main__":
    service = PredictionServicePhase4E()
    res = service.predict_date("2024-11-16")
    print(res.model_dump_json(indent=2))
