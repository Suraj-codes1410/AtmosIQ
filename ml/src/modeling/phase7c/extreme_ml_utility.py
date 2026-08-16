"""
AtmosIQ Phase 7C: Extreme Event ML Utility Evaluator (Workstream I).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ExtremeMLUtilityEvaluator:
    """Evaluates ML forecasting errors specifically on extreme held-out test observations."""

    def __init__(self):
        self.thresholds = [100.0, 150.0, 200.0, 250.0]

    def evaluate_extreme_ml_utility(
        self,
        df_real_test: pd.DataFrame,
        predictions_dict: Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        y_true = df_real_test["pm25"].values
        records = []

        for th in self.thresholds:
            mask = (y_true >= th)
            n_samples = int(mask.sum())
            if n_samples == 0:
                continue

            y_sub = y_true[mask]
            for model_name, preds in predictions_dict.items():
                p_sub = preds[mask]
                mae = float(mean_absolute_error(y_sub, p_sub))
                rmse = float(np.sqrt(mean_squared_error(y_sub, p_sub)))
                bias = float(np.mean(p_sub - y_sub))

                records.append({
                    "threshold_ug_m3": th,
                    "model_experiment": model_name,
                    "test_sample_count": n_samples,
                    "mae": mae,
                    "rmse": rmse,
                    "mean_bias": bias,
                })

        df_ext_ml = pd.DataFrame(records)

        # Compare extreme 250+ MAE for real_only vs augmented
        sub_250_real = df_ext_ml[(df_ext_ml["threshold_ug_m3"] == 250.0) & (df_ext_ml["model_experiment"] == "real_only")]
        sub_250_aug = df_ext_ml[(df_ext_ml["threshold_ug_m3"] == 250.0) & (df_ext_ml["model_experiment"] == "real_plus_synthetic_100")]

        r_250_mae = float(sub_250_real["mae"].iloc[0]) if len(sub_250_real) > 0 else 0.0
        a_250_mae = float(sub_250_aug["mae"].iloc[0]) if len(sub_250_aug) > 0 else 0.0

        summary = {
            "extreme_250_real_only_mae": r_250_mae,
            "extreme_250_augmented_mae": a_250_mae,
            "delta_extreme_250_mae": a_250_mae - r_250_mae,
            "extreme_utility_status": "PASS" if a_250_mae <= r_250_mae + 2.0 else "WARNING",
        }

        return df_ext_ml, summary
