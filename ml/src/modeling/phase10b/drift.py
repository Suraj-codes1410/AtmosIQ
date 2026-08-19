"""
AtmosIQ Phase 10B: Feature Drift, Prediction Drift, and Physical Sanity Monitoring Engine.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import logging

logger = logging.getLogger(__name__)


class Phase10BDriftMonitor:
    """Calculates PSI, KS, Wasserstein distance, prediction drift, and physical sanity metrics."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    @staticmethod
    def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10, eps: float = 1e-4) -> float:
        """Calculates Population Stability Index (PSI) between baseline and target distributions."""
        b_clean = baseline[~np.isnan(baseline)]
        t_clean = target[~np.isnan(target)]

        if len(b_clean) < 10 or len(t_clean) < 10:
            return 0.0

        # Create quantile bins from baseline
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(b_clean, quantiles)
        bin_edges = np.unique(bin_edges) # Avoid duplicate edges for constant values

        if len(bin_edges) < 2:
            return 0.0

        b_counts, _ = np.histogram(b_clean, bins=bin_edges)
        t_counts, _ = np.histogram(t_clean, bins=bin_edges)

        b_pct = (b_counts / len(b_clean)) + eps
        t_pct = (t_counts / len(t_clean)) + eps

        b_pct /= np.sum(b_pct)
        t_pct /= np.sum(t_pct)

        psi = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
        return float(psi)

    def monitor_feature_drift(
        self,
        df_baseline: pd.DataFrame,
        df_current: pd.DataFrame
    ) -> pd.DataFrame:
        """Audits all 35 prediction-safe features for distribution drift."""
        records = []

        for feat in self.feature_registry:
            if feat not in df_baseline.columns or feat not in df_current.columns:
                continue

            b_vals = df_baseline[feat].dropna().values
            c_vals = df_current[feat].dropna().values

            if len(b_vals) == 0 or len(c_vals) == 0:
                continue

            b_mean, b_std = float(np.mean(b_vals)), float(np.std(b_vals))
            c_mean, c_std = float(np.mean(c_vals)), float(np.std(c_vals))

            # PSI
            psi_val = self.calculate_psi(b_vals, c_vals)

            # KS & Wasserstein
            ks_stat, ks_pval = ks_2samp(b_vals, c_vals)
            scale = b_std if b_std > 1e-6 else 1.0
            w_dist = float(wasserstein_distance(b_vals / scale, c_vals / scale))

            # Severity tier
            if psi_val < 0.10 and w_dist < 0.20:
                severity = "GREEN"
            elif psi_val < 0.25 and w_dist < 0.40:
                severity = "YELLOW"
            elif psi_val < 0.50 and w_dist < 0.70:
                severity = "ORANGE"
            else:
                severity = "RED"

            missing_rate = float(df_current[feat].isna().mean())

            records.append({
                "feature_name": feat,
                "baseline_mean": b_mean,
                "current_mean": c_mean,
                "baseline_std": b_std,
                "current_std": c_std,
                "psi": psi_val,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "normalized_wasserstein_dist": w_dist,
                "missing_rate": missing_rate,
                "drift_severity": severity,
            })

        df_res = pd.DataFrame(records)
        return df_res.sort_values(by="normalized_wasserstein_dist", ascending=False).reset_index(drop=True)

    def monitor_prediction_drift(
        self,
        baseline_preds: np.ndarray,
        current_preds: np.ndarray
    ) -> pd.DataFrame:
        """Audits model forecast distribution changes over time."""
        b_clean = baseline_preds[~np.isnan(baseline_preds)]
        c_clean = current_preds[~np.isnan(current_preds)]

        psi_pred = self.calculate_psi(b_clean, c_clean)
        ks_stat, ks_pval = ks_2samp(b_clean, c_clean)
        scale = np.std(b_clean) if np.std(b_clean) > 1e-6 else 1.0
        w_dist = float(wasserstein_distance(b_clean / scale, c_clean / scale))

        record = {
            "baseline_mean": float(np.mean(b_clean)),
            "current_mean": float(np.mean(c_clean)),
            "baseline_median": float(np.median(b_clean)),
            "current_median": float(np.median(c_clean)),
            "baseline_std": float(np.std(b_clean)),
            "current_std": float(np.std(c_clean)),
            "current_p10": float(np.percentile(c_clean, 10)),
            "current_p50": float(np.percentile(c_clean, 50)),
            "current_p90": float(np.percentile(c_clean, 90)),
            "current_p99": float(np.percentile(c_clean, 99)),
            "fraction_high_pm25_ge_250": float(np.mean(c_clean >= 250.0)),
            "prediction_psi": psi_pred,
            "prediction_ks_stat": float(ks_stat),
            "prediction_wasserstein_dist": w_dist,
            "prediction_drift_status": "NORMAL" if psi_pred < 0.25 else "MATERIAL_DRIFT",
        }

        return pd.DataFrame([record])

    def monitor_physical_sanity(self, df_current: pd.DataFrame) -> pd.DataFrame:
        """Audits physical sanity checks across atmospheric variables."""
        checks = []

        # 1. Non-negative PM2.5
        if "pm25" in df_current.columns:
            neg_count = int((df_current["pm25"] < 0).sum())
            checks.append({
                "variable": "pm25",
                "physical_rule": "pm25 >= 0 µg/m³",
                "violation_count": neg_count,
                "violation_rate": float(neg_count / len(df_current)),
                "status": "PASS" if neg_count == 0 else "FAIL_NEGATIVE_PM25",
            })

        # 2. Relative Humidity in [0, 100]
        if "relative_humidity_2m" in df_current.columns:
            rh_invalid = int(((df_current["relative_humidity_2m"] < 0) | (df_current["relative_humidity_2m"] > 100)).sum())
            checks.append({
                "variable": "relative_humidity_2m",
                "physical_rule": "RH in [0, 100] %",
                "violation_count": rh_invalid,
                "violation_rate": float(rh_invalid / len(df_current)),
                "status": "PASS" if rh_invalid == 0 else "FAIL_OUT_OF_BOUNDS_RH",
            })

        # 3. Non-negative Wind Speed
        if "wind_speed_10m" in df_current.columns:
            ws_neg = int((df_current["wind_speed_10m"] < 0).sum())
            checks.append({
                "variable": "wind_speed_10m",
                "physical_rule": "wind_speed >= 0 m/s",
                "violation_count": ws_neg,
                "violation_rate": float(ws_neg / len(df_current)),
                "status": "PASS" if ws_neg == 0 else "FAIL_NEGATIVE_WIND",
            })

        # 4. Non-negative Boundary Layer Height
        if "boundary_layer_height" in df_current.columns:
            pblh_neg = int((df_current["boundary_layer_height"] < 0).sum())
            checks.append({
                "variable": "boundary_layer_height",
                "physical_rule": "PBLH >= 0 m",
                "violation_count": pblh_neg,
                "violation_rate": float(pblh_neg / len(df_current)),
                "status": "PASS" if pblh_neg == 0 else "FAIL_NEGATIVE_PBLH",
            })

        # 5. Ventilation Index Identity: VI ≈ ws * PBLH
        if all(c in df_current.columns for c in ["ventilation_index", "wind_speed_10m", "boundary_layer_height"]):
            vi_expected = df_current["wind_speed_10m"] * df_current["boundary_layer_height"]
            vi_diff = np.abs(df_current["ventilation_index"] - vi_expected)
            vi_mismatch = int((vi_diff > 1.0).sum())
            checks.append({
                "variable": "ventilation_index",
                "physical_rule": "VI ≈ ws * PBLH (±1.0 m²/s)",
                "violation_count": vi_mismatch,
                "violation_rate": float(vi_mismatch / len(df_current)),
                "status": "PASS" if vi_mismatch == 0 else "WARNING_VI_DISCREPANCY",
            })

        return pd.DataFrame(checks)
