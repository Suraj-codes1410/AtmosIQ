import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("SeasonalAnalysisPhase4H")


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


class SeasonalAnalysisPhase4H:
    """
    Seasonal Performance Breakdown Engine for Phase 4H.
    Evaluates out-of-sample predictions across Winter, Summer, Monsoon, and Post-Monsoon.
    """

    def run_seasonal_eval(self, model_predictions: dict) -> pd.DataFrame:
        """
        model_predictions: dict mapping model_key -> DataFrame with columns ['date', 'y_true', 'y_pred']
        """
        logger.info("Executing Seasonal Performance Evaluation...")
        records = []

        for model_key, pred_df in model_predictions.items():
            df = pred_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['season'] = df['date'].dt.month.apply(get_season)

            for season in ["Winter", "Summer", "Monsoon", "Post-Monsoon"]:
                sub = df[df['season'] == season]
                if len(sub) == 0:
                    continue

                y_true = sub['y_true'].values
                y_pred = sub['y_pred'].values

                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(root_mean_squared_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                parts = model_key.split("__")
                m_name = parts[0]
                f_set = parts[1] if len(parts) > 1 else "default"

                records.append({
                    "model_key": model_key,
                    "model_name": m_name,
                    "feature_set": f_set,
                    "season": season,
                    "sample_size": len(sub),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4)
                })

        return pd.DataFrame(records)
