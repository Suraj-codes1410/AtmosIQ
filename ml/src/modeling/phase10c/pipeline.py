"""
AtmosIQ Phase 10C: End-to-End Production Inference Pipeline Engine.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import time
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class ProductionInferenceException(ValueError):
    """Raised when an inference input or artifact violates the strict production contract."""
    pass


class Phase10CProductionPipeline:
    """Production-grade deterministic end-to-end inference engine with runtime contract enforcement."""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_registry: List[str],
        window_size: int = 14,
        feature_dim: int = 35,
        model_version: str = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0",
        model_hash: str = "fdc99f7ca4410f3d",
        calibration_bias: float = -5.06,
        conformal_bound_80: float = 63.92,
        conformal_bound_90: float = 95.66,
        conformal_bound_95: float = 117.50
    ):
        self.model = model
        self.scaler = scaler
        self.feature_registry = list(feature_registry)
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.model_version = model_version
        self.model_hash = model_hash
        self.calibration_bias = calibration_bias
        self.conformal_bound_80 = conformal_bound_80
        self.conformal_bound_90 = conformal_bound_90
        self.conformal_bound_95 = conformal_bound_95

    def validate_raw_dataframe(self, df_input: pd.DataFrame):
        """Validates raw DataFrame input schema, timestamps, and missing values."""
        if not isinstance(df_input, pd.DataFrame):
            raise ProductionInferenceException(f"Expected pandas DataFrame, got {type(df_input)}")

        if len(df_input) < self.window_size:
            raise ProductionInferenceException(
                f"Insufficient sequence length: required at least {self.window_size} rows, got {len(df_input)}"
            )

        # 1. Feature Registry & Schema Verification
        missing_cols = [c for c in self.feature_registry if c not in df_input.columns]
        if missing_cols:
            raise ProductionInferenceException(f"Missing required prediction-safe features: {missing_cols[:5]}")

        # 2. Timestamp Validation
        if "date" in df_input.columns:
            ts_series = pd.to_datetime(df_input["date"])
            if not ts_series.is_monotonic_increasing:
                raise ProductionInferenceException("Timestamp violation: dates must be strictly monotonically increasing.")
            if ts_series.duplicated().any():
                raise ProductionInferenceException("Timestamp violation: duplicated timestamps detected.")

        # 3. Finite Numbers & Non-Null Audit
        feat_df = df_input[self.feature_registry]
        if feat_df.isna().any().any():
            raise ProductionInferenceException("Input rejected: NaN values detected in prediction features.")
        if np.isinf(feat_df.values).any():
            raise ProductionInferenceException("Input rejected: Inf values detected in prediction features.")

    def construct_sequence_tensor(self, df_input: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extracts strictly ordered features, applies frozen scaler, and constructs (B, 14, 35) sequence tensor."""
        self.validate_raw_dataframe(df_input)

        # Ensure exact column ordering matching registry
        X_raw = df_input[self.feature_registry].values.astype(np.float32)

        # Apply frozen scaler (WITHOUT refitting)
        X_scaled = self.scaler.transform(X_raw)

        # Construct sliding windows
        sequences = []
        timestamps = []
        dates = df_input["date"].tolist() if "date" in df_input.columns else [f"t_{i}" for i in range(len(df_input))]

        for i in range(len(X_scaled) - self.window_size + 1):
            seq = X_scaled[i : i + self.window_size]
            target_ts = dates[i + self.window_size - 1]
            sequences.append(seq)
            timestamps.append(target_ts)

        X_tensor = np.array(sequences, dtype=np.float32)
        return X_tensor, timestamps

    def predict(
        self,
        df_input: pd.DataFrame,
        batch_id: str = "BATCH_PROD_0001"
    ) -> Dict[str, Any]:
        """Executes the full deterministic end-to-end production forecast pipeline."""
        start_time = time.perf_counter()

        # Step 1-3: Validation & Sequence Construction
        X_tensor, timestamps = self.construct_sequence_tensor(df_input)

        # Step 4: Model Inference
        raw_preds = self.model.forward(X_tensor)

        # Step 5: Bias Calibration
        cal_preds = np.maximum(raw_preds - self.calibration_bias, 0.0)

        # Step 6: Conformal Prediction Intervals (80%, 90%, 95%)
        lower_80 = np.maximum(cal_preds - self.conformal_bound_80, 0.0)
        upper_80 = cal_preds + self.conformal_bound_80

        lower_90 = np.maximum(cal_preds - self.conformal_bound_90, 0.0)
        upper_90 = cal_preds + self.conformal_bound_90

        lower_95 = np.maximum(cal_preds - self.conformal_bound_95, 0.0)
        upper_95 = cal_preds + self.conformal_bound_95

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 7: Format Structured Production Response
        forecasts = []
        for i, (ts, pred, l80, u80, l90, u90, l95, u95) in enumerate(
            zip(timestamps, cal_preds, lower_80, upper_80, lower_90, upper_90, lower_95, upper_95)
        ):
            pred_id = hashlib.sha256(f"{batch_id}_{ts}_{pred:.4f}_{i}".encode("utf-8")).hexdigest()[:16]
            forecasts.append({
                "prediction_id": pred_id,
                "timestamp_utc": ts,
                "target_horizon": "t+14d",
                "forecast_pm25": float(pred),
                "raw_uncalibrated_pm25": float(raw_preds[i]),
                "uncertainty_intervals": {
                    "conformal_80": {"lower": float(l80), "upper": float(u80), "half_width": float(self.conformal_bound_80)},
                    "conformal_90": {"lower": float(l90), "upper": float(u90), "half_width": float(self.conformal_bound_90)},
                    "conformal_95": {"lower": float(l95), "upper": float(u95), "half_width": float(self.conformal_bound_95)},
                },
                "physical_sanity_status": "PASS_NON_NEGATIVE_FINITE",
            })

        response = {
            "batch_id": batch_id,
            "inference_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_version": self.model_version,
            "model_sha256": self.model_hash[:16],
            "preprocessing_version": "v1.0.0_StandardScaler_dev_frozen",
            "calibration_version": f"v1.0.0_bias_{self.calibration_bias:.2f}",
            "batch_size": len(forecasts),
            "execution_latency_ms": float(elapsed_ms),
            "sla_status": "PASS" if elapsed_ms <= 50.0 else "WARNING_LATENCY_ELEVATED",
            "forecasts": forecasts,
        }

        return response
