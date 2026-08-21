"""
AtmosIQ Phase 11B: Operational Baseline & Distribution Analysis Engine.

Audits input quality, feature drift, prediction distribution, and calibration/conformal
uncertainty behavior across baseline development data vs controlled operational replay.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import logging

from .config import (
    PRODUCTION_FEATURES,
    PSI_GREEN_THRESHOLD,
    PSI_YELLOW_THRESHOLD,
    WASSERSTEIN_GREEN_MAX,
    CERTIFIED_BOUND_90,
    CERTIFIED_CALIBRATION_BIAS,
)
from ml.src.modeling.phase10b.drift import Phase10BDriftMonitor

logger = logging.getLogger(__name__)


class Phase11BBaselineEngine:
    """Calculates operational baseline metrics and audits distribution shifts."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.drift_monitor = Phase10BDriftMonitor(PRODUCTION_FEATURES)
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Loads and partitions dataset into baseline (2020-2021) and operational replay (2022-2024)."""
        df = pd.read_csv(self.dataset_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            self.df_baseline = df[df["date"] <= "2021-12-31"].copy().reset_index(drop=True)
            self.df_replay   = df[df["date"] >= "2022-01-01"].copy().reset_index(drop=True)
        else:
            n_split = int(len(df) * 0.4)
            self.df_baseline = df.iloc[:n_split].copy().reset_index(drop=True)
            self.df_replay   = df.iloc[n_split:].copy().reset_index(drop=True)

    def audit_input_quality(self) -> pd.DataFrame:
        """Audits input quality metrics on the operational replay stream."""
        records = []
        total_rows = len(self.df_replay)

        for col in PRODUCTION_FEATURES:
            if col in self.df_replay.columns:
                series = self.df_replay[col]
                missing_cnt = int(series.isna().sum())
                inf_cnt     = int(np.isinf(series.to_numpy()).sum())
                valid_cnt   = total_rows - missing_cnt - inf_cnt
                records.append({
                    "feature_name": col,
                    "total_sequences_observed": total_rows,
                    "missing_count": missing_cnt,
                    "infinite_count": inf_cnt,
                    "valid_count": valid_cnt,
                    "input_quality_status": "PASS_CLEAN" if (missing_cnt == 0 and inf_cnt == 0) else "ALERT_DIRTY",
                })
            else:
                records.append({
                    "feature_name": col,
                    "total_sequences_observed": total_rows,
                    "missing_count": total_rows,
                    "infinite_count": 0,
                    "valid_count": 0,
                    "input_quality_status": "FAIL_MISSING_COLUMN",
                })

        return pd.DataFrame(records)

    def compute_feature_monitoring(self) -> pd.DataFrame:
        """Computes PSI, Wasserstein distance, KS stat, and drift status for all 35 features."""
        return self.drift_monitor.monitor_feature_drift(self.df_baseline, self.df_replay)

    def compute_prediction_monitoring(self, predictions_baseline: np.ndarray, predictions_replay: np.ndarray) -> Dict[str, Any]:
        """Computes prediction distribution summary, PSI, and extreme event fractions."""
        p_base = np.array(predictions_baseline)
        p_rep  = np.array(predictions_replay)

        psi = self.drift_monitor.calculate_psi(p_base, p_rep)
        w_dist = float(wasserstein_distance(p_base, p_rep))
        ks_stat, ks_pval = ks_2samp(p_base, p_rep)

        # Extreme prediction threshold (> 250 µg/m³)
        base_extreme_pct = float(np.mean(p_base > 250.0) * 100.0)
        rep_extreme_pct  = float(np.mean(p_rep > 250.0) * 100.0)

        drift_status = "GREEN_NO_DRIFT"
        if psi > PSI_YELLOW_THRESHOLD or w_dist > WASSERSTEIN_GREEN_MAX:
            drift_status = "YELLOW_MODERATE_DRIFT"
        if psi > 0.40:
            drift_status = "ORANGE_SIGNIFICANT_DRIFT"

        summary = {
            "baseline_count": len(p_base),
            "replay_count": len(p_rep),
            "baseline_mean": float(np.mean(p_base)),
            "baseline_std": float(np.std(p_base)),
            "baseline_median": float(np.median(p_base)),
            "baseline_min": float(np.min(p_base)),
            "baseline_max": float(np.max(p_base)),
            "baseline_extreme_pct_gt250": base_extreme_pct,
            "replay_mean": float(np.mean(p_rep)),
            "replay_std": float(np.std(p_rep)),
            "replay_median": float(np.median(p_rep)),
            "replay_min": float(np.min(p_rep)),
            "replay_max": float(np.max(p_rep)),
            "replay_extreme_pct_gt250": rep_extreme_pct,
            "prediction_psi": float(psi),
            "prediction_wasserstein_distance": float(w_dist),
            "prediction_ks_stat": float(ks_stat),
            "prediction_ks_pvalue": float(ks_pval),
            "prediction_drift_status": drift_status,
        }
        return summary

    def compute_calibration_uncertainty_monitoring(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Audits empirical 90% conformal coverage, interval width, and residual bias on replay stream."""
        residuals = y_pred - y_true
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        bias = float(np.mean(residuals))

        # Conformal interval coverage check
        lower_90 = np.maximum(y_pred - CERTIFIED_BOUND_90, 0.0)
        upper_90 = y_pred + CERTIFIED_BOUND_90
        covered = (y_true >= lower_90) & (y_true <= upper_90)
        empirical_coverage = float(np.mean(covered) * 100.0)

        coverage_status = "PASS_WITHIN_TARGET" if empirical_coverage >= 88.0 else "WARNING_UNDERCOVERAGE"

        return {
            "evaluation_samples": len(y_true),
            "mae_pm25": round(mae, 3),
            "rmse_pm25": round(rmse, 3),
            "bias_pm25": round(bias, 3),
            "certified_bias_offset": CERTIFIED_CALIBRATION_BIAS,
            "conformal_half_width_90": CERTIFIED_BOUND_90,
            "target_coverage_pct": 90.0,
            "empirical_coverage_pct": round(empirical_coverage, 2),
            "coverage_status": coverage_status,
        }
