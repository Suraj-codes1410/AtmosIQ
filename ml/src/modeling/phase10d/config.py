"""
AtmosIQ Phase 10D: Configuration System for Final Production Release & Go-Live Gate.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class Phase10DConfig:
    phase_name: str = "Phase 10D"
    phase_version: str = "1.0.0"
    target_variable: str = "pm25"
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2025])
    default_seed: int = 42
    extreme_threshold: float = 250.0 # µg/m³

    # Canonical Production Candidate & Promoted Release Identity
    candidate_model_id: str = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0"
    production_release_id: str = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0"
    fallback_model_id: str = "AtmosIQ_DL_LSTM_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0"
    stress_test_model_id: str = "AtmosIQ_DL_TCN_CAL07_50_RESEARCH_v1.0.0"
    previous_production_version: str = "MODEL_V3_PRODUCTION"

    production_architecture: str = "TCN"
    production_parameters_count: int = 849
    sequence_window: int = 14
    feature_dim: int = 35
    production_augmentation_ratio: float = 0.25
    stress_test_augmentation_ratio: float = 0.50

    # Calibration & Conformal Bounds
    calibration_bias: float = -5.06 # µg/m³
    conformal_bound_80: float = 63.92 # µg/m³
    conformal_bound_90: float = 95.66 # µg/m³
    conformal_bound_95: float = 117.50 # µg/m³

    # Date Partitions
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Latency & Resource SLAs
    sla_single_inference_ms: float = 10.0
    sla_batch_inference_ms: float = 50.0
    sla_max_memory_mb: float = 256.0
    sla_min_throughput_hz: float = 1000.0

    # Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))

    phase9_checkpoints_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9_deep_learning/checkpoints"))
    phase10_benchmarks_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10_production/benchmarks"))
    phase10b_manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10b_observability/manifests"))
    phase10c_benchmarks_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10c_inference/benchmarks"))

    # Output Directories
    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release"))
    bundle_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release/release_bundle"))
    manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release/manifests"))
    audits_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release/audits"))
    benchmarks_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release/benchmarks"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release/reports"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release/figures"))
    hashes_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase10d_release/hashes"))

    def get_release_hash(self) -> str:
        cfg = {
            "phase_name": self.phase_name,
            "production_release_id": self.production_release_id,
            "candidate_model_id": self.candidate_model_id,
            "sequence_window": self.sequence_window,
            "feature_dim": self.feature_dim,
            "calibration_bias": self.calibration_bias,
            "conformal_bound_90": self.conformal_bound_90,
            "sla_single_ms": self.sla_single_inference_ms,
        }
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
