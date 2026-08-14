import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates MAE, RMSE, R2, Median AE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "median_ae": round(medae, 4)
    }


def calculate_metric_deltas(candidate_metrics: dict, control_metrics: dict) -> dict:
    """
    Calculates deltas relative to control:
    - delta_mae = MAE_v3 - MAE_v2 (Negative = Improvement)
    - delta_rmse = RMSE_v3 - RMSE_v2 (Negative = Improvement)
    - delta_r2 = R2_v3 - R2_v2 (Positive = Improvement)
    - delta_median_ae = MedAE_v3 - MedAE_v2 (Negative = Improvement)
    """
    delta_mae = candidate_metrics["mae"] - control_metrics["mae"]
    delta_rmse = candidate_metrics["rmse"] - control_metrics["rmse"]
    delta_r2 = candidate_metrics["r2"] - control_metrics["r2"]
    delta_medae = candidate_metrics["median_ae"] - control_metrics["median_ae"]

    return {
        "delta_mae": round(delta_mae, 4),
        "delta_rmse": round(delta_rmse, 4),
        "delta_r2": round(delta_r2, 4),
        "delta_median_ae": round(delta_medae, 4)
    }
