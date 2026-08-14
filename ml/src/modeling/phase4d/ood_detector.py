import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("OODDetectorPhase4D")


class OODDetectorPhase4D:
    """
    AtmosIQ Phase 4D Out-Of-Distribution (OOD) Detector.
    Evaluates standardized distance and feature percentile bounds to identify environmentally unrealistic or OOD counterfactual scenarios.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.means = None
        self.stds = None

    def fit_reference_distribution(self, X_ref: pd.DataFrame, feature_order: list):
        """Fits reference distribution mean and std for all features."""
        self.feature_order = feature_order
        X_mat = X_ref[feature_order].values
        self.means = np.mean(X_mat, axis=0)
        self.stds = np.std(X_mat, axis=0) + 1e-8
        logger.info("OOD detector reference distribution fitted.")

    def evaluate_ood(self, x_cf: np.ndarray) -> tuple[bool, float, str]:
        """
        Calculates standardized distance score for x_cf.
        Returns (ood_flag, ood_score, ood_reason).
        """
        z_scores = np.abs((x_cf - self.means) / self.stds)
        max_z = float(np.max(z_scores))
        mean_z = float(np.mean(z_scores))

        # Thresholds: max z-score > 4.5 or mean z-score > 2.0 indicates OOD
        if max_z > 4.5 or mean_z > 2.0:
            ood_flag = True
            max_feat_idx = int(np.argmax(z_scores))
            max_feat_name = self.feature_order[max_feat_idx]
            ood_reason = f"Extreme z-score ({max_z:.2f}) on feature '{max_feat_name}'"
        else:
            ood_flag = False
            ood_reason = "Normal feature distribution"

        ood_score = float(max_z)
        return ood_flag, ood_score, ood_reason


if __name__ == "__main__":
    detector = OODDetectorPhase4D()
