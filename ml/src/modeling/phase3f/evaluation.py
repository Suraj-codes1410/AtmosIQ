import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from ml.src.utils.logger import setup_logger

logger = setup_logger("EvaluationPhase3F")


class MetricsEvaluatorPhase3F:
    """
    AtmosIQ Phase 3F Metrics & Overfitting Evaluator.
    Calculates MAE, RMSE, R2, Median AE, and Train-to-Evaluation Generalization Gaps.
    """

    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        """Calculates MAE, RMSE, R2, and Median AE."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        med_ae = float(median_absolute_error(y_true, y_pred))

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Median_AE": round(med_ae, 4)
        }

    @staticmethod
    def calculate_overfitting_metrics(m_train: dict[str, float], m_eval: dict[str, float]) -> dict[str, float]:
        """Calculates Train-to-Eval R2 Gap and MAE Ratio."""
        r2_gap = round(m_train["R2"] - m_eval["R2"], 4)
        mae_ratio = round(m_eval["MAE"] / (m_train["MAE"] + 1e-5), 4)
        return {
            "Train_R2": m_train["R2"],
            "Eval_R2": m_eval["R2"],
            "R2_Gap": r2_gap,
            "Train_MAE": m_train["MAE"],
            "Eval_MAE": m_eval["MAE"],
            "MAE_Ratio": mae_ratio
        }
