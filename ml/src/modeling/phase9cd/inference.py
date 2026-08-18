"""
AtmosIQ Phase 9D: Inference Engine, Runtime Contract Validator & Profiling Suite.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
import logging

logger = logging.getLogger(__name__)


class InferenceContractViolation(ValueError):
    """Raised when an inference input violates the strict Phase 9D inference contract."""
    pass


class Phase9DInferenceEngine:
    """Production-grade deterministic inference engine with runtime contract enforcement."""

    def __init__(
        self,
        model: Any,
        feature_registry: List[str],
        window_size: int = 14,
        feature_dim: int = 35,
        model_version: str = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0",
        calibration_bias: float = 0.0,
        interval_bound_90: float = 25.0
    ):
        self.model = model
        self.feature_registry = list(feature_registry)
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.model_version = model_version
        self.calibration_bias = calibration_bias
        self.interval_bound_90 = interval_bound_90

    def validate_input_tensor(self, X: np.ndarray):
        """Strictly validates input tensor against contract constraints."""
        if not isinstance(X, np.ndarray):
            raise InferenceContractViolation(f"Input must be numpy ndarray, got {type(X)}")

        if X.ndim != 3:
            raise InferenceContractViolation(f"Expected 3D tensor (Batch, W={self.window_size}, D={self.feature_dim}), got ndim={X.ndim}")

        B, W, D = X.shape
        if W != self.window_size:
            raise InferenceContractViolation(f"Sequence length mismatch: expected {self.window_size}, got {W}")

        if D != self.feature_dim:
            raise InferenceContractViolation(f"Feature dimension mismatch: expected {self.feature_dim}, got {D}")

        if np.isnan(X).any():
            raise InferenceContractViolation("Inference rejected: input tensor contains NaN values.")

        if np.isinf(X).any():
            raise InferenceContractViolation("Inference rejected: input tensor contains Inf values.")

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """Runs validated deterministic inference and returns point forecasts with calibrated prediction intervals."""
        self.validate_input_tensor(X)

        raw_pred = self.model.forward(X)
        cal_pred = np.maximum(raw_pred - self.calibration_bias, 0.0)

        response = {
            "model_version": self.model_version,
            "forecast_pm25": cal_pred.tolist(),
            "batch_size": len(cal_pred),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if return_uncertainty:
            lower = np.maximum(cal_pred - self.interval_bound_90, 0.0)
            upper = cal_pred + self.interval_bound_90
            response["uncertainty_interval_90"] = {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "half_width": float(self.interval_bound_90),
                "method": "Conformal Residual Interval",
            }

        return response

    def profile_latency(self, X_sample: np.ndarray, n_iterations: int = 50) -> Dict[str, float]:
        """Profiles single-item and batch inference latency and throughput."""
        self.validate_input_tensor(X_sample)
        single_x = X_sample[:1]

        # Warmup
        _ = self.model.forward(single_x)
        _ = self.model.forward(X_sample)

        # Single item latency
        start_single = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.model.forward(single_x)
        single_lat_ms = ((time.perf_counter() - start_single) / n_iterations) * 1000.0

        # Batch latency
        start_batch = time.perf_counter()
        for _ in range(n_iterations):
            _ = self.model.forward(X_sample)
        batch_lat_ms = ((time.perf_counter() - start_batch) / n_iterations) * 1000.0

        throughput = (len(X_sample) * n_iterations) / (time.perf_counter() - start_batch)

        return {
            "single_item_latency_ms": float(single_lat_ms),
            "batch_latency_ms": float(batch_lat_ms),
            "batch_size": len(X_sample),
            "throughput_samples_per_sec": float(throughput),
        }

    def run_robustness_test_suite(self, X_valid: np.ndarray) -> pd.DataFrame:
        """Executes controlled adversarial and malformed input tests to verify safe rejection."""
        self.validate_input_tensor(X_valid)
        test_cases = [
            ("NaN Value in Tensor", lambda: self.predict(np.where(np.isnan(X_valid), 0.0, np.nan))),
            ("Inf Value in Tensor", lambda: self.predict(np.where(X_valid == X_valid, np.inf, 0.0))),
            ("Wrong Sequence Length (W=7)", lambda: self.predict(X_valid[:, :7, :])),
            ("Wrong Sequence Length (W=21)", lambda: self.predict(np.pad(X_valid, ((0, 0), (0, 7), (0, 0))))),
            ("Wrong Feature Dimension (D=30)", lambda: self.predict(X_valid[:, :, :30])),
            ("Wrong Feature Dimension (D=40)", lambda: self.predict(np.pad(X_valid, ((0, 0), (0, 0), (0, 5))))),
            ("2D Tensor Dimension (B, D)", lambda: self.predict(X_valid[:, 0, :])),
            ("Non-Numpy Object Input", lambda: self.predict([[1.0] * 35] * 14)),
        ]

        audit_results = []
        for name, test_fn in test_cases:
            try:
                test_fn()
                status = "FAIL_UNSAFE_ACCEPTED"
                rejected_cleanly = False
            except (InferenceContractViolation, ValueError, TypeError) as e:
                status = "PASS_SAFELY_REJECTED"
                rejected_cleanly = True

            audit_results.append({
                "test_case": name,
                "safely_rejected": rejected_cleanly,
                "audit_status": status,
            })

        return pd.DataFrame(audit_results)
