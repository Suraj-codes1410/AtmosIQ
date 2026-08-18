"""
AtmosIQ Phase 9C: Model Hardening, Calibration, Uncertainty & Explainability Engine.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew, kurtosis
import json
import logging

logger = logging.getLogger(__name__)


class Phase9CHardener:
    """Implements research-grade model calibration, conformal prediction intervals, residual diagnostics, and explainability."""

    def __init__(self, feature_names: List[str], extreme_threshold: float = 250.0):
        self.feature_names = feature_names
        self.extreme_threshold = extreme_threshold
        self.calibration_bias: float = 0.0
        self.conformal_q80: float = 0.0
        self.conformal_q90: float = 0.0
        self.conformal_q95: float = 0.0

    def fit_calibration_and_uncertainty(self, y_val_true: np.ndarray, y_val_pred: np.ndarray):
        """Fits calibration bias and conformal prediction interval quantiles strictly on development validation data."""
        residuals = y_val_pred - y_val_true
        # Mean bias correction parameter
        self.calibration_bias = float(np.mean(residuals))

        # Absolute error quantiles for conformal prediction intervals
        abs_errors = np.abs(residuals)
        self.conformal_q80 = float(np.quantile(abs_errors, 0.80))
        self.conformal_q90 = float(np.quantile(abs_errors, 0.90))
        self.conformal_q95 = float(np.quantile(abs_errors, 0.95))

        logger.info(f"Fitted validation calibration bias: {self.calibration_bias:.2f} µg/m³")
        logger.info(f"Fitted conformal error bounds: 80% ±{self.conformal_q80:.2f}, 90% ±{self.conformal_q90:.2f}, 95% ±{self.conformal_q95:.2f} µg/m³")

    def calibrate_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """Applies fitted linear bias correction and clamps non-negativity."""
        cal_pred = y_pred - self.calibration_bias
        return np.maximum(cal_pred, 0.0)

    def compute_prediction_intervals(self, y_pred: np.ndarray, alpha: float = 0.10) -> Tuple[np.ndarray, np.ndarray, float]:
        """Computes empirical conformal prediction intervals [lower, upper] and half-width."""
        if alpha <= 0.05:
            bound = self.conformal_q95
        elif alpha <= 0.10:
            bound = self.conformal_q90
        else:
            bound = self.conformal_q80

        lower = np.maximum(y_pred - bound, 0.0)
        upper = y_pred + bound
        return lower, upper, bound

    def evaluate_uncertainty_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> Dict[str, float]:
        """Calculates interval coverage, average width, and extreme-event coverage."""
        covered = (y_true >= lower) & (y_true <= upper)
        coverage_rate = float(np.mean(covered))
        avg_width = float(np.mean(upper - lower))

        # Extreme subset coverage
        extreme_mask = (y_true >= self.extreme_threshold)
        if np.any(extreme_mask):
            ext_coverage = float(np.mean(covered[extreme_mask]))
        else:
            ext_coverage = coverage_rate

        return {
            "interval_coverage": coverage_rate,
            "average_interval_width": avg_width,
            "extreme_interval_coverage": ext_coverage,
            "uncertainty_type": "Empirical Conformal (Aleatoric Residual)",
        }

    def compute_residual_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Computes statistical moments, autocorrelation, and heteroscedasticity indicators."""
        residuals = y_pred - y_true
        abs_res = np.abs(residuals)

        # Lag-1 Autocorrelation
        if len(residuals) > 2:
            r1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        else:
            r1 = 0.0

        # Heteroscedasticity correlation: corr(abs_residual, y_true)
        if len(y_true) > 2 and np.std(y_true) > 1e-6:
            r_het, _ = pearsonr(y_true, abs_res)
            het_corr = float(r_het)
        else:
            het_corr = 0.0

        return {
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_skew": float(skew(residuals)) if len(residuals) > 2 else 0.0,
            "residual_kurtosis": float(kurtosis(residuals)) if len(residuals) > 2 else 0.0,
            "lag1_autocorrelation": r1,
            "heteroscedasticity_corr": het_corr,
            "max_error": float(np.max(abs_res)),
        }

    def compute_permutation_explainability(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_repeats: int = 3,
        seed: int = 42
    ) -> pd.DataFrame:
        """Computes feature permutation importance on the evaluation sequence tensor (N, W, D)."""
        base_pred = model.forward(X_test)
        base_mae = float(np.mean(np.abs(base_pred - y_test)))

        importance_records = []
        B, W, D = X_test.shape
        np.random.seed(seed)

        for d_idx, feat_name in enumerate(self.feature_names):
            perm_maes = []
            for _ in range(n_repeats):
                X_perm = X_test.copy()
                # Permute across batch dimension for feature d_idx across all time steps
                perm_idx = np.random.permutation(B)
                X_perm[:, :, d_idx] = X_perm[perm_idx, :, d_idx]

                pred_perm = model.forward(X_perm)
                perm_mae = float(np.mean(np.abs(pred_perm - y_test)))
                perm_maes.append(perm_mae - base_mae)

            mean_importance = float(np.mean(perm_maes))
            std_importance = float(np.std(perm_maes))

            importance_records.append({
                "feature_name": feat_name,
                "feature_index": d_idx,
                "importance_mae_delta": mean_importance,
                "importance_std": std_importance,
            })

        df_imp = pd.DataFrame(importance_records)
        df_imp = df_imp.sort_values(by="importance_mae_delta", ascending=False).reset_index(drop=True)
        df_imp["rank"] = np.arange(1, len(df_imp) + 1)
        return df_imp
