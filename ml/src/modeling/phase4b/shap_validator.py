import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("SHAPValidatorPhase4B")


class SHAPValidatorPhase4B:
    """
    AtmosIQ Phase 4B SHAP Additivity & Reconstruction Validator.
    Verifies that sum(SHAP) + base_value == model_prediction within numerical tolerance.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4b", tolerance: float = 1e-4):
        self.exp_dir = Path(exp_dir)
        self.summary_dir = self.exp_dir / "summaries"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.tolerance = tolerance

    def validate_reconstruction(self, base_value: float, shap_matrix: np.ndarray, predictions: np.ndarray, group_shap_matrix: np.ndarray = None) -> dict:
        """Performs strict additivity verification for feature-level and group-level SHAP values."""
        logger.info("Executing TreeSHAP additivity verification (base_value + sum(SHAP) == predicted_pm25)...")

        feature_sum = shap_matrix.sum(axis=1)
        reconstructed_preds = base_value + feature_sum
        errors = np.abs(predictions - reconstructed_preds)

        max_err = float(np.max(errors))
        mean_err = float(np.mean(errors))
        med_err = float(np.median(errors))
        p95_err = float(np.percentile(errors, 95))
        rmse_err = float(np.sqrt(np.mean(errors ** 2)))

        logger.info(f"Feature Additivity Results -> Max Err: {max_err:.4e}, Mean Err: {mean_err:.4e}, P95 Err: {p95_err:.4e}")

        if max_err > self.tolerance:
            raise ValueError(f"CRITICAL ADDITIVITY FAILURE: Maximum SHAP reconstruction error {max_err:.4e} exceeds tolerance {self.tolerance:.4e}!")

        # Group reconstruction check
        group_max_err, group_mean_err = 0.0, 0.0
        if group_shap_matrix is not None:
            group_sum = group_shap_matrix.sum(axis=1)
            group_reconstructed = base_value + group_sum
            group_errors = np.abs(predictions - group_reconstructed)
            group_max_err = float(np.max(group_errors))
            group_mean_err = float(np.mean(group_errors))
            logger.info(f"Group Additivity Results -> Max Err: {group_max_err:.4e}, Mean Err: {group_mean_err:.4e}")
            if group_max_err > self.tolerance:
                raise ValueError(f"CRITICAL GROUP ADDITIVITY FAILURE: Group reconstruction error {group_max_err:.4e} exceeds tolerance!")

        # Export reconstruction_summary.csv
        summary_rows = [
            {"metric": "max_absolute_error", "value": max_err, "status": "PASS" if max_err <= self.tolerance else "FAIL"},
            {"metric": "mean_absolute_error", "value": mean_err, "status": "PASS"},
            {"metric": "median_absolute_error", "value": med_err, "status": "PASS"},
            {"metric": "p95_absolute_error", "value": p95_err, "status": "PASS"},
            {"metric": "rmse_reconstruction", "value": rmse_err, "status": "PASS"},
            {"metric": "group_max_absolute_error", "value": group_max_err, "status": "PASS"},
            {"metric": "tolerance_threshold", "value": self.tolerance, "status": "INFO"}
        ]

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.summary_dir / "reconstruction_summary.csv", index=False)

        logger.info(f"SHAP reconstruction additivity verification 100% PASS (Max Error: {max_err:.4e} <= {self.tolerance}).")

        return {
            "max_error": max_err,
            "mean_error": mean_err,
            "median_error": med_err,
            "p95_error": p95_err,
            "rmse_error": rmse_err,
            "passed": max_err <= self.tolerance
        }


if __name__ == "__main__":
    validator = SHAPValidatorPhase4B()
