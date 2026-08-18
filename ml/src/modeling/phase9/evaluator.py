"""
AtmosIQ Phase 9: Comprehensive Performance Evaluator.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew
import logging

logger = logging.getLogger(__name__)


class Phase9Evaluator:
    """Evaluates temporal deep-learning models across standard, extreme, annual, seasonal, and regime subsets."""

    def __init__(self, extreme_threshold: float = 250.0):
        self.extreme_threshold = extreme_threshold

    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates standard regression metrics, Pearson correlation, and extreme-event MAE."""
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        residuals = y_pred - y_true
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        if len(y_true) > 1 and np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
            r_val, _ = pearsonr(y_true, y_pred)
            r_corr = float(r_val)
        else:
            r_corr = 0.0

        # Extreme events subset
        extreme_mask = (y_true >= self.extreme_threshold)
        if np.any(extreme_mask):
            extreme_mae = float(np.mean(np.abs(y_pred[extreme_mask] - y_true[extreme_mask])))
            extreme_rmse = float(np.sqrt(np.mean((y_pred[extreme_mask] - y_true[extreme_mask]) ** 2)))
            extreme_count = int(np.sum(extreme_mask))
        else:
            extreme_mae = mae
            extreme_rmse = rmse
            extreme_count = 0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": r_corr,
            "extreme_mae": extreme_mae,
            "extreme_rmse": extreme_rmse,
            "extreme_count": extreme_count,
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(np.std(y_pred)),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_skew": float(skew(residuals)) if len(residuals) > 2 else 0.0,
            "max_abs_error": float(np.max(np.abs(residuals))),
        }

    def evaluate_temporal_breakdowns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: List[str]
    ) -> Dict[str, Any]:
        """Calculates breakdown metrics by Year and Season."""
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "y_true": y_true,
            "y_pred": y_pred,
        })
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        def assign_season(m):
            if m in [12, 1, 2]: return "Winter"
            elif m in [3, 4, 5]: return "Summer" # Indian pre-monsoon summer
            elif m in [6, 7, 8, 9]: return "Monsoon"
            else: return "Post-Monsoon"

        df["season"] = df["month"].apply(assign_season)

        annual_results = {}
        for yr, df_yr in df.groupby("year"):
            annual_results[str(yr)] = self.evaluate_metrics(df_yr["y_true"].values, df_yr["y_pred"].values)

        seasonal_results = {}
        for s, df_s in df.groupby("season"):
            seasonal_results[s] = self.evaluate_metrics(df_s["y_true"].values, df_s["y_pred"].values)

        return {
            "annual": annual_results,
            "seasonal": seasonal_results,
        }
