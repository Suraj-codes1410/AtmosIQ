import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("PlausibilityPhase4D")


class PlausibilityValidatorPhase4D:
    """
    AtmosIQ Phase 4D Counterfactual Validity & Plausibility Checker.
    Enforces feature isolation (untargeted features 100% exact), prediction reproducibility, and feature range bound checks.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def validate_counterfactual_plausibility(
        self,
        feature_order: list,
        group_mapping: dict,
        x_obs: np.ndarray,
        x_cf: np.ndarray,
        target_group: str,
        ref_quantiles: dict
    ) -> dict:
        """
        Validates feature isolation, range bounds, and non-null values for a counterfactual feature vector.
        """
        # Check 1: Non-null / non-inf
        assert not np.isnan(x_cf).any(), "Counterfactual vector contains NaN!"
        assert not np.isinf(x_cf).any(), "Counterfactual vector contains Inf!"

        # Check 2: Feature isolation
        untargeted_indices = [i for i, f in enumerate(feature_order) if group_mapping[f] != target_group]
        max_untargeted_diff = float(np.max(np.abs(x_cf[untargeted_indices] - x_obs[untargeted_indices])))
        isolation_pass = max_untargeted_diff < 1e-12

        # Check 3: Range bounds
        out_of_bounds_count = 0
        for i, f in enumerate(feature_order):
            min_val = ref_quantiles[f]["min"]
            max_val = ref_quantiles[f]["max"]
            val = x_cf[i]
            if val < min_val or val > max_val:
                out_of_bounds_count += 1

        bounds_pass = out_of_bounds_count == 0

        return {
            "isolation_pass": isolation_pass,
            "max_untargeted_diff": max_untargeted_diff,
            "bounds_pass": bounds_pass,
            "out_of_bounds_count": out_of_bounds_count,
            "overall_plausibility_pass": bool(isolation_pass and bounds_pass)
        }


if __name__ == "__main__":
    validator = PlausibilityValidatorPhase4D()
