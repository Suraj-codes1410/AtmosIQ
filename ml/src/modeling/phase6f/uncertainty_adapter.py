import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6f.config import DecisionSupportConfigPhase6F

logger = setup_logger("UncertaintyAdapterPhase6F")


class UncertaintyAdapterPhase6F:
    """
    Production Prediction Interval Adapter for Phase 6F.
    Consumes the frozen Phase 6D Normalized Heteroscedastic Conformal Prediction layer.
    """

    def __init__(self, config: DecisionSupportConfigPhase6F, unc_dir: str = "ml/uncertainty/production/v1"):
        self.config = config
        self.unc_dir = Path(unc_dir)
        self.calib_path = self.unc_dir / "calibration_artifacts.json"
        assert self.calib_path.exists(), f"Missing calibration artifacts at {self.calib_path}"
        
        with open(self.calib_path, "r") as f:
            self.calib_data = json.load(f)

        self.regime_scales = self.calib_data["regime_scales_ugm3"]
        self.quantiles = self.calib_data["calibrated_quantiles"]
        self.method_name = self.calib_data["method"]
        self.method_version = self.config.production_uncertainty_version

    def get_regime_name(self, pred_val: float) -> str:
        if pred_val < 60.0:
            return "Low"
        elif pred_val < 120.0:
            return "Moderate"
        elif pred_val < 250.0:
            return "High"
        else:
            return "Extreme"

    def compute_prediction_interval(self, prediction: float, nominal_coverage: float = 0.90) -> Dict[str, Any]:
        """
        Computes calibrated prediction interval using normalized_conformal.
        Enforces physical non-negativity (lower_bound >= 0.0).
        """
        if nominal_coverage not in self.config.supported_coverage_levels:
            raise ValueError(f"Unsupported nominal coverage: {nominal_coverage}. Supported: {self.config.supported_coverage_levels}")

        cov_key = f"{int(nominal_coverage * 100)}pct_nominal"
        if cov_key not in self.quantiles:
            raise KeyError(f"Quantile key {cov_key} missing from calibration artifacts!")

        q_val = self.quantiles[cov_key]
        regime = self.get_regime_name(prediction)
        sigma = self.regime_scales[regime]

        half_width = q_val * sigma
        lower_bound = max(0.0, float(prediction - half_width))
        upper_bound = float(prediction + half_width)
        interval_width = upper_bound - lower_bound

        return {
            "prediction": float(prediction),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interval_width": interval_width,
            "nominal_coverage": float(nominal_coverage),
            "pollution_regime": regime,
            "regime_scale_sigma": float(sigma),
            "conformal_quantile": float(q_val),
            "method": self.method_name,
            "method_version": self.method_version,
            "unit": "µg/m³"
        }
