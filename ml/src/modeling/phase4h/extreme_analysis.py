import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ExtremeAnalysisPhase4H")


class ExtremeAnalysisPhase4H:
    """
    Extreme Pollution Event Evaluation Engine for Phase 4H.
    Evaluates out-of-sample prediction performance on high-pollution regimes:
    1. Extreme Pollution Days (PM2.5 >= 150 ug/m3)
    2. Top 10% Pollution Days (>= 90th percentile)
    3. Stubble Peak Episode (November 1–30)
    """

    def run_extreme_eval(self, model_predictions: dict) -> pd.DataFrame:
        """
        model_predictions: dict mapping model_key -> DataFrame with columns ['date', 'y_true', 'y_pred']
        """
        logger.info("Executing Extreme Pollution Event Evaluation...")
        records = []

        for model_key, pred_df in model_predictions.items():
            df = pred_df.copy()
            df['date'] = pd.to_datetime(df['date'])

            p90_threshold = float(np.percentile(df['y_true'].values, 90))

            subsets = {
                "Extreme_PM25_gte_150": df[df['y_true'] >= 150.0],
                "Top_10pct_Pollution_Days": df[df['y_true'] >= p90_threshold],
                "November_Stubble_Peak": df[(df['date'].dt.month == 11)]
            }

            parts = model_key.split("__")
            m_name = parts[0]
            f_set = parts[1] if len(parts) > 1 else "default"

            for regime_name, sub in subsets.items():
                if len(sub) == 0:
                    continue

                y_true = sub['y_true'].values
                y_pred = sub['y_pred'].values

                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(root_mean_squared_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
                mean_bias = float(np.mean(y_pred - y_true))

                records.append({
                    "model_key": model_key,
                    "model_name": m_name,
                    "feature_set": f_set,
                    "regime": regime_name,
                    "sample_count": len(sub),
                    "mean_actual_pm25": round(float(np.mean(y_true)), 2),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                    "mean_bias": round(mean_bias, 4)
                })

        return pd.DataFrame(records)
