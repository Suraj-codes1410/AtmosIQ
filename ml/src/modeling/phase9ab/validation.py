"""
AtmosIQ Phase 9B: Independent Candidate Validation & Residual Diagnostic Engine.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew, kurtosis
import json
import logging

logger = logging.getLogger(__name__)


class Phase9BValidator:
    """Performs deep multi-dimensional independent validation on certified research candidate models."""

    def __init__(self, extreme_threshold: float = 250.0):
        self.extreme_threshold = extreme_threshold

    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates standard regression and diagnostic metrics."""
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

        # Extreme subset
        extreme_mask = (y_true >= self.extreme_threshold)
        if np.any(extreme_mask):
            ext_true = y_true[extreme_mask]
            ext_pred = y_pred[extreme_mask]
            extreme_mae = float(np.mean(np.abs(ext_pred - ext_true)))
            extreme_rmse = float(np.sqrt(np.mean((ext_pred - ext_true) ** 2)))
            extreme_count = int(np.sum(extreme_mask))
            underpred_rate = float(np.mean(ext_pred < ext_true))
            overpred_rate = float(np.mean(ext_pred > ext_true))
            ext_bias = float(np.mean(ext_pred - ext_true))
        else:
            extreme_mae, extreme_rmse, extreme_count = mae, rmse, 0
            underpred_rate, overpred_rate, ext_bias = 0.0, 0.0, 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": r_corr,
            "extreme_mae": extreme_mae,
            "extreme_rmse": extreme_rmse,
            "extreme_count": extreme_count,
            "extreme_underpred_rate": underpred_rate,
            "extreme_overpred_rate": overpred_rate,
            "extreme_bias": ext_bias,
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(np.std(y_pred)),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_skew": float(skew(residuals)) if len(residuals) > 2 else 0.0,
            "residual_kurtosis": float(kurtosis(residuals)) if len(residuals) > 2 else 0.0,
            "max_abs_error": float(np.max(np.abs(residuals))),
            "physical_validity_rate": float(np.mean(y_pred >= 0.0)),
            "negative_predictions_count": int(np.sum(y_pred < 0.0)),
            "nan_count": int(np.sum(np.isnan(y_pred))),
            "inf_count": int(np.sum(np.isinf(y_pred))),
        }

    def evaluate_yearly_breakdowns(self, y_true: np.ndarray, y_pred: np.ndarray, dates: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({"date": pd.to_datetime(dates), "y_true": y_true, "y_pred": y_pred})
        df["year"] = df["date"].dt.year
        records = []
        for yr, df_yr in df.groupby("year"):
            m = self.evaluate_metrics(df_yr["y_true"].values, df_yr["y_pred"].values)
            records.append({"year": yr, "observations": len(df_yr), **m})
        return pd.DataFrame(records)

    def evaluate_seasonal_breakdowns(self, y_true: np.ndarray, y_pred: np.ndarray, dates: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({"date": pd.to_datetime(dates), "y_true": y_true, "y_pred": y_pred})
        df["month"] = df["date"].dt.month
        def assign_season(m):
            if m in [12, 1, 2]: return "Winter"
            elif m in [3, 4, 5]: return "Summer"
            elif m in [6, 7, 8, 9]: return "Monsoon"
            else: return "Post-Monsoon"
        df["season"] = df["month"].apply(assign_season)
        records = []
        season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
        for sn in season_order:
            df_sn = df[df["season"] == sn]
            if len(df_sn) > 0:
                m = self.evaluate_metrics(df_sn["y_true"].values, df_sn["y_pred"].values)
                records.append({"season": sn, "observations": len(df_sn), **m})
        return pd.DataFrame(records)

    def evaluate_regime_breakdowns(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        def get_regime(val):
            if val <= 30: return "Good"
            elif val <= 60: return "Satisfactory"
            elif val <= 120: return "Moderate"
            elif val <= 250: return "Poor/Severe"
            else: return "Emergency"
        df["regime"] = df["y_true"].apply(get_regime)
        reg_order = ["Good", "Satisfactory", "Moderate", "Poor/Severe", "Emergency"]
        records = []
        for reg in reg_order:
            df_r = df[df["regime"] == reg]
            if len(df_r) > 0:
                m = self.evaluate_metrics(df_r["y_true"].values, df_r["y_pred"].values)
                records.append({"regime": reg, "observations": len(df_r), **m})
        return pd.DataFrame(records)

    def extract_failure_cases(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: List[str],
        top_n: int = 25
    ) -> pd.DataFrame:
        df = pd.DataFrame({
            "timestamp": dates,
            "observed_pm25": y_true,
            "predicted_pm25": y_pred,
            "residual": y_pred - y_true,
            "absolute_error": np.abs(y_pred - y_true),
        })
        df["date_dt"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["date_dt"].dt.year
        df["month"] = df["date_dt"].dt.month
        def assign_season(m):
            if m in [12, 1, 2]: return "Winter"
            elif m in [3, 4, 5]: return "Summer"
            elif m in [6, 7, 8, 9]: return "Monsoon"
            else: return "Post-Monsoon"
        def get_regime(val):
            if val <= 30: return "Good"
            elif val <= 60: return "Satisfactory"
            elif val <= 120: return "Moderate"
            elif val <= 250: return "Poor/Severe"
            else: return "Emergency"
        df["season"] = df["month"].apply(assign_season)
        df["regime"] = df["observed_pm25"].apply(get_regime)
        df["extreme_event"] = df["observed_pm25"] >= self.extreme_threshold
        df["trajectory_id"] = [f"EVAL_TRAJ_{i//14:04d}" for i in range(len(df))]

        cols = [
            "timestamp", "observed_pm25", "predicted_pm25", "absolute_error",
            "season", "year", "regime", "extreme_event", "trajectory_id"
        ]
        return df.sort_values(by="absolute_error", ascending=False).head(top_n)[cols]
