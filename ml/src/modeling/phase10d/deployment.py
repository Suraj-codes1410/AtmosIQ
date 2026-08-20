"""
AtmosIQ Phase 10D: Deployed Production Service & API Contract Validator.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

from .config import Phase10DConfig
from ml.src.modeling.phase9.models import Phase9TCNModel
from ml.src.modeling.phase9.trainer import Phase9Trainer

logger = logging.getLogger(__name__)


class ServiceContractException(ValueError):
    """Raised when an API request violates the deployed service contract."""
    pass


class Phase10DDeploymentService:
    """Production service simulation loading directly from an isolated release bundle."""

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        self._is_ready = False
        self._load_from_bundle()

    def _load_from_bundle(self):
        """Loads and initializes all model and runtime components from the release bundle."""
        # 1. Load Model Config & Checkpoint
        with open(self.bundle_dir / "model_config.json") as f:
            self.model_config = json.load(f)

        self.model = Phase9TCNModel(
            window_size=self.model_config["sequence_window"],
            feature_dim=self.model_config["feature_dimension"],
            seed=2025
        )
        trainer = Phase9Trainer(self.model, seed=2025)
        trainer.load_checkpoint(self.bundle_dir / "model_checkpoint.json", self.model)

        # 2. Load Feature Registry
        feat_df = pd.read_csv(self.bundle_dir / "feature_registry.csv")
        self.feature_registry = feat_df["feature_name"].tolist()

        # 3. Reconstruct Scaler
        with open(self.bundle_dir / "scaler_state.json") as f:
            scaler_data = json.load(f)
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(scaler_data["mean"], dtype=np.float64)
        self.scaler.scale_ = np.array(scaler_data["scale"], dtype=np.float64)
        self.scaler.var_ = np.array(scaler_data["var"], dtype=np.float64)
        self.scaler.n_features_in_ = scaler_data["n_features_in"]

        # 4. Load Calibration & Uncertainty
        with open(self.bundle_dir / "calibration_params.json") as f:
            self.cal_params = json.load(f)
        self.calibration_bias = float(self.cal_params["bias_offset_pm25"])

        with open(self.bundle_dir / "uncertainty_config.json") as f:
            self.unc_config = json.load(f)
        self.bound_80 = float(self.unc_config["bounds"]["conformal_80"]["half_width"])
        self.bound_90 = float(self.unc_config["bounds"]["conformal_90"]["half_width"])
        self.bound_95 = float(self.unc_config["bounds"]["conformal_95"]["half_width"])

        self._is_ready = True

    def health_endpoint(self) -> Dict[str, Any]:
        """Health check endpoint: returns operational status."""
        return {
            "status": "HEALTHY",
            "service": "AtmosIQ_Production_Service",
            "model_loaded": self.model is not None,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def readiness_endpoint(self) -> Dict[str, Any]:
        """Readiness check endpoint: verifies service is ready to process inference traffic."""
        return {
            "status": "READY" if self._is_ready else "NOT_READY",
            "model_version": self.model_config["model_id"],
            "feature_count": len(self.feature_registry),
            "scaler_ready": hasattr(self.scaler, "mean_"),
            "calibration_ready": self.calibration_bias is not None,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def version_endpoint(self) -> Dict[str, Any]:
        """Model version and lineage metadata endpoint."""
        return {
            "model_id": self.model_config["model_id"],
            "candidate_id": self.model_config["candidate_id"],
            "architecture": self.model_config["architecture"],
            "parameters": self.model_config["parameter_count"],
            "model_sha256": self.model_config["model_sha256"][:16],
            "release_status": "RELEASE_CERTIFIED",
        }

    def predict_endpoint(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Inference endpoint accepting structured JSON payload or DataFrame rows."""
        start_time = time.perf_counter()

        if "records" not in payload or not isinstance(payload["records"], list):
            raise ServiceContractException("Malformed payload: 'records' list required.")

        records = payload["records"]
        if len(records) < self.model_config["sequence_window"]:
            raise ServiceContractException(
                f"Insufficient sequence length: required at least {self.model_config['sequence_window']} rows, got {len(records)}"
            )

        df_input = pd.DataFrame(records)

        # 1. Validate Schema & Features
        missing_cols = [c for c in self.feature_registry if c not in df_input.columns]
        if missing_cols:
            raise ServiceContractException(f"Missing required prediction-safe features: {missing_cols[:5]}")

        # 2. Validate Timestamps
        if "date" in df_input.columns:
            ts_series = pd.to_datetime(df_input["date"])
            if not ts_series.is_monotonic_increasing:
                raise ServiceContractException("Timestamp violation: dates must be strictly monotonic.")
            if ts_series.duplicated().any():
                raise ServiceContractException("Timestamp violation: duplicate dates detected.")

        # 3. Finite Numbers & Non-Null
        feat_df = df_input[self.feature_registry]
        if feat_df.isna().any().any():
            raise ServiceContractException("Input rejected: NaN values in features.")
        if np.isinf(feat_df.values).any():
            raise ServiceContractException("Input rejected: Inf values in features.")

        # 4. Construct Tensor & Scale
        X_raw = feat_df.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)

        sequences = []
        timestamps = []
        dates = df_input["date"].tolist() if "date" in df_input.columns else [f"t_{i}" for i in range(len(df_input))]

        w = self.model_config["sequence_window"]
        for i in range(len(X_scaled) - w + 1):
            seq = X_scaled[i : i + w]
            target_ts = dates[i + w - 1]
            sequences.append(seq)
            timestamps.append(target_ts)

        X_tensor = np.array(sequences, dtype=np.float32)

        # 5. Forward Model Inference
        raw_preds = self.model.forward(X_tensor)

        # 6. Calibration & Conformal Bounds
        cal_preds = np.maximum(raw_preds - self.calibration_bias, 0.0)

        lower_90 = np.maximum(cal_preds - self.bound_90, 0.0)
        upper_90 = cal_preds + self.bound_90

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        forecasts = []
        for i, (ts, pred, l90, u90) in enumerate(zip(timestamps, cal_preds, lower_90, upper_90)):
            pred_id = hashlib.sha256(f"{ts}_{pred:.4f}_{i}".encode("utf-8")).hexdigest()[:16]
            forecasts.append({
                "prediction_id": pred_id,
                "timestamp_utc": ts,
                "forecast_pm25": float(pred),
                "lower_90": float(l90),
                "upper_90": float(u90),
                "conformal_half_width": float(self.bound_90),
            })

        return {
            "status": "SUCCESS",
            "model_version": self.model_config["model_id"],
            "execution_latency_ms": float(elapsed_ms),
            "batch_size": len(forecasts),
            "forecasts": forecasts,
        }
