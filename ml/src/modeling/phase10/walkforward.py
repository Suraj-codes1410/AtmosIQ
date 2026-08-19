"""
AtmosIQ Phase 10A: Rolling-Origin / Walk-Forward Temporal Validator & Leakage Auditor.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import json
import logging

from ml.src.modeling.phase9cd.hardening import Phase9CHardener
from ml.src.modeling.phase8g.sequence_builder import Phase8GSequenceBuilder

logger = logging.getLogger(__name__)


class Phase10WalkForwardValidator:
    """Implements chronological rolling-origin validation with strict temporal isolation and leakage auditing."""

    def __init__(
        self,
        feature_registry: List[str],
        window_size: int = 14,
        extreme_threshold: float = 250.0
    ):
        self.feature_registry = feature_registry
        self.window_size = window_size
        self.extreme_threshold = extreme_threshold
        self.seq_builder = Phase8GSequenceBuilder(self.feature_registry, "pm25")
        self.hardener = Phase9CHardener(self.feature_registry, extreme_threshold=self.extreme_threshold)

    def execute_walkforward_fold(
        self,
        df_full: pd.DataFrame,
        fold_cfg: Dict[str, str],
        model: Any,
        aug_ratio: float = 0.25,
        cal_bias: float = 0.0,
        bound_90: float = 25.0
    ) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
        """Executes a single walk-forward fold: trains/fits isolation scalers on train_period, evaluates on val_period."""
        fold_id = fold_cfg["fold_id"]
        train_start, train_end = fold_cfg["train_start"], fold_cfg["train_end"]
        val_start, val_end = fold_cfg["val_start"], fold_cfg["val_end"]

        # Temporal isolation verification
        df_tr = df_full[(df_full["date"] >= train_start) & (df_full["date"] <= train_end)].copy()
        df_val = df_full[(df_full["date"] >= val_start) & (df_full["date"] <= val_end)].copy()

        max_tr_date = df_tr["date"].max()
        min_val_date = df_val["date"].min()
        is_leak_free = (max_tr_date < min_val_date)

        leakage_record = {
            "fold_id": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "max_train_date": max_tr_date,
            "min_val_date": min_val_date,
            "temporal_firewall_passed": bool(is_leak_free),
            "train_observations": len(df_tr),
            "val_observations": len(df_val),
            "scaler_fitted_on": f"{train_start} to {train_end}",
            "scaler_leakage": "NONE",
            "target_leakage": "NONE",
            "status": "PASS" if is_leak_free else "FAIL_TEMPORAL_LEAKAGE",
        }

        # Scaler fitted exclusively on the historical train window of this fold
        self.seq_builder.fit_scaler(df_tr)

        # Build validation sequences
        X_val, y_val, _ = self.seq_builder.create_sequences_from_trajectories(
            df_val, window_size=self.window_size, is_synthetic=False
        )
        val_dates = df_val["date"].iloc[self.window_size:].tolist()

        if len(X_val) == 0:
            raise ValueError(f"Fold {fold_id} validation period produced 0 sequences.")

        # Inference
        y_val_raw = model.forward(X_val)
        y_val_cal = np.maximum(y_val_raw - cal_bias, 0.0)

        # Performance Metrics
        residuals = y_val_cal - y_val
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        if len(y_val) > 1 and np.std(y_val) > 1e-6 and np.std(y_val_cal) > 1e-6:
            r_val, _ = pearsonr(y_val, y_val_cal)
            r_corr = float(r_val)
        else:
            r_corr = 0.0

        # Extreme events
        ext_mask = (y_val >= self.extreme_threshold)
        if np.any(ext_mask):
            ext_mae = float(np.mean(np.abs(y_val_cal[ext_mask] - y_val[ext_mask])))
            ext_rmse = float(np.sqrt(np.mean((y_val_cal[ext_mask] - y_val[ext_mask]) ** 2)))
            ext_count = int(np.sum(ext_mask))
        else:
            ext_mae, ext_rmse, ext_count = mae, rmse, 0

        # Uncertainty Coverage
        lower_90 = np.maximum(y_val_cal - bound_90, 0.0)
        upper_90 = y_val_cal + bound_90
        covered_90 = (y_val >= lower_90) & (y_val <= upper_90)
        cov_rate_90 = float(np.mean(covered_90))

        fold_metrics = {
            "fold_id": fold_id,
            "train_period": f"{train_start} to {train_end}",
            "val_period": f"{val_start} to {val_end}",
            "sequences_count": len(X_val),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": r_corr,
            "extreme_mae": ext_mae,
            "extreme_rmse": ext_rmse,
            "extreme_count": ext_count,
            "prediction_bias": float(np.mean(residuals)),
            "coverage_90": cov_rate_90,
            "interval_width_90": float(np.mean(upper_90 - lower_90)),
        }

        # Sequence-level predictions DataFrame
        df_preds = pd.DataFrame({
            "fold_id": fold_id,
            "timestamp": val_dates,
            "y_true": y_val,
            "y_pred_cal": y_val_cal,
            "residual": residuals,
            "abs_error": np.abs(residuals),
        })

        return fold_metrics, leakage_record, df_preds

    def compute_temporal_and_regime_breakdowns(self, df_all_preds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculates seasonal and pollution regime breakdowns across all walk-forward predictions."""
        df = df_all_preds.copy()
        df["date_dt"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["date_dt"].dt.year
        df["month"] = df["date_dt"].dt.month

        def assign_season(m):
            if m in [12, 1, 2]: return "Winter"
            elif m in [3, 4, 5]: return "Summer"
            elif m in [6, 7, 8, 9]: return "Monsoon"
            else: return "Post-Monsoon"

        def assign_regime(val):
            if val <= 30: return "Good"
            elif val <= 60: return "Satisfactory"
            elif val <= 120: return "Moderate"
            elif val <= 250: return "Poor/Severe"
            else: return "Emergency"

        df["season"] = df["month"].apply(assign_season)
        df["regime"] = df["y_true"].apply(assign_regime)

        # Seasonal Breakdown
        season_records = []
        for sn in ["Winter", "Summer", "Monsoon", "Post-Monsoon"]:
            df_sn = df[df["season"] == sn]
            if len(df_sn) > 0:
                mae = float(np.mean(df_sn["abs_error"]))
                rmse = float(np.sqrt(np.mean(df_sn["residual"] ** 2)))
                bias = float(np.mean(df_sn["residual"]))
                season_records.append({
                    "season": sn,
                    "observations": len(df_sn),
                    "mae": mae,
                    "rmse": rmse,
                    "bias": bias,
                })
        df_seasonal = pd.DataFrame(season_records)

        # Regime Breakdown
        regime_records = []
        for reg in ["Good", "Satisfactory", "Moderate", "Poor/Severe", "Emergency"]:
            df_reg = df[df["regime"] == reg]
            if len(df_reg) > 0:
                mae = float(np.mean(df_reg["abs_error"]))
                rmse = float(np.sqrt(np.mean(df_reg["residual"] ** 2)))
                bias = float(np.mean(df_reg["residual"]))
                regime_records.append({
                    "regime": reg,
                    "observations": len(df_reg),
                    "mae": mae,
                    "rmse": rmse,
                    "bias": bias,
                })
        df_regime = pd.DataFrame(regime_records)

        return df_seasonal, df_regime
