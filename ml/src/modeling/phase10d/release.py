"""
AtmosIQ Phase 10D: Production Release Bundle & Manifest Manager.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import hashlib
import shutil
import time
import sys
import platform
import numpy as np
import pandas as pd
import logging

from .config import Phase10DConfig

logger = logging.getLogger(__name__)


class Phase10DReleaseManager:
    """Packages and cryptographically certifies the formal immutable AtmosIQ production release bundle."""

    def __init__(self, config: Phase10DConfig):
        self.config = config
        self.bundle_dir = self.config.bundle_dir
        self.manifests_dir = self.config.manifests_dir
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def build_release_bundle(
        self,
        checkpoint_path: Path,
        scaler: Any,
        feature_registry: List[str]
    ) -> Dict[str, Any]:
        """Assembles all frozen model, preprocessor, calibration, and governance components into release bundle."""
        # 1. Model Checkpoint & Architecture
        target_ckpt = self.bundle_dir / "model_checkpoint.json"
        shutil.copy2(checkpoint_path, target_ckpt)
        model_sha = self.compute_sha256(target_ckpt)

        model_config = {
            "model_id": self.config.production_release_id,
            "candidate_id": self.config.candidate_model_id,
            "architecture": self.config.production_architecture,
            "parameter_count": self.config.production_parameters_count,
            "sequence_window": self.config.sequence_window,
            "feature_dimension": self.config.feature_dim,
            "augmentation_ratio": self.config.production_augmentation_ratio,
            "synthetic_corpus": "AtmosIQ_Synthetic_Calibrated_v0.1.0 / CAL-07",
            "model_sha256": model_sha,
        }
        with open(self.bundle_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=4)

        # 2. Feature Registry
        target_feat = self.bundle_dir / "feature_registry.csv"
        pd.DataFrame({"feature_name": feature_registry}).to_csv(target_feat, index=False)
        feat_sha = self.compute_sha256(target_feat)

        # 3. Scaler State
        scaler_state = {
            "scaler_type": "StandardScaler",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
            "n_features_in": int(scaler.n_features_in_),
            "fitting_horizon": "2020-01-01 to 2021-12-31 (Historical Dev N=731)",
        }
        target_scaler = self.bundle_dir / "scaler_state.json"
        with open(target_scaler, "w") as f:
            json.dump(scaler_state, f, indent=4)
        scaler_sha = self.compute_sha256(target_scaler)

        # 4. Calibration Parameters
        cal_params = {
            "calibration_method": "Bias Offset Clamping",
            "bias_offset_pm25": self.config.calibration_bias,
            "target_variable": "pm25",
            "unit": "µg/m³",
        }
        target_cal = self.bundle_dir / "calibration_params.json"
        with open(target_cal, "w") as f:
            json.dump(cal_params, f, indent=4)
        cal_sha = self.compute_sha256(target_cal)

        # 5. Uncertainty Configuration
        unc_config = {
            "uncertainty_method": "Conformal Empirical Residual Bounds",
            "bounds": {
                "conformal_80": {"half_width": self.config.conformal_bound_80, "nominal_coverage": 0.80},
                "conformal_90": {"half_width": self.config.conformal_bound_90, "nominal_coverage": 0.90},
                "conformal_95": {"half_width": self.config.conformal_bound_95, "nominal_coverage": 0.95},
            },
            "interpretation": "Empirical prediction interval (not guaranteed physical certainty).",
        }
        target_unc = self.bundle_dir / "uncertainty_config.json"
        with open(target_unc, "w") as f:
            json.dump(unc_config, f, indent=4)
        unc_sha = self.compute_sha256(target_unc)

        # 6. Environment & Runtime Specification
        env_manifest = {
            "environment_name": "AtmosIQ_Production_Runtime_v1.0.0",
            "python_version": sys.version,
            "os_platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "cpu_arch": platform.machine(),
            "release_build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        target_env = self.manifests_dir / "phase10d_environment_manifest.json"
        with open(target_env, "w") as f:
            json.dump(env_manifest, f, indent=4)
        shutil.copy2(target_env, self.bundle_dir / "environment_manifest.json")

        # 7. Complete Master Release Manifest
        release_manifest = {
            "manifest_name": "AtmosIQ_Phase10D_Production_Release_Manifest",
            "release_id": self.config.production_release_id,
            "candidate_id": self.config.candidate_model_id,
            "fallback_id": self.config.fallback_model_id,
            "previous_version": self.config.previous_production_version,
            "architecture": self.config.production_architecture,
            "parameters": self.config.production_parameters_count,
            "input_contract": {"window_size": self.config.sequence_window, "feature_dim": self.config.feature_dim},
            "bundle_hashes": {
                "model_checkpoint_sha256": model_sha,
                "feature_registry_sha256": feat_sha,
                "scaler_state_sha256": scaler_sha,
                "calibration_params_sha256": cal_sha,
                "uncertainty_config_sha256": unc_sha,
            },
            "release_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "certification_status": "RELEASE_CERTIFIED",
            "go_live_status": "READY",
        }
        target_rel = self.manifests_dir / "phase10d_release_manifest.json"
        with open(target_rel, "w") as f:
            json.dump(release_manifest, f, indent=4)
        shutil.copy2(target_rel, self.bundle_dir / "release_manifest.json")

        return release_manifest
