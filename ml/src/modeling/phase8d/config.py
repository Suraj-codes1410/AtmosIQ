"""
AtmosIQ Phase 8D: Calibration Configuration System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class CalibrationConfigPhase8D:
    phase_name: str = "Phase 8D"
    phase_version: str = "1.0.0"
    baseline_corpus_version: str = "AtmosIQ_Synthetic_Production_v1.0.0"
    calibrated_corpus_version: str = "AtmosIQ_Synthetic_Calibrated_v0.1.0"
    global_seed: int = 42

    # Partitions
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Calibration Strategies
    candidates: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"id": "CAL-00", "name": "No Calibration (Baseline Phase 8C)", "type": "baseline"},
        {"id": "CAL-01", "name": "Distribution Calibration (Wasserstein Matching)", "type": "distribution"},
        {"id": "CAL-02", "name": "Regime Calibration (Seasonal & Regime Alignment)", "type": "regime"},
        {"id": "CAL-03", "name": "Temporal Calibration (ACF & Persistence Filtering)", "type": "temporal"},
        {"id": "CAL-04", "name": "Multivariate Calibration (Frobenius Correlation)", "type": "multivariate"},
        {"id": "CAL-05", "name": "OOD-Aware Calibration (Density & Boundary Truncation)", "type": "ood"},
        {"id": "CAL-06", "name": "Extreme-Tail Calibration (Tightened Risk Filters)", "type": "extreme_tail"},
        {"id": "CAL-07", "name": "Combined Multi-Objective Calibration", "type": "combined"},
    ])

    # Horizons & Augmentation
    approved_horizons: List[int] = field(default_factory=lambda: [14, 30])
    recommended_augmentation_ratio: float = 0.25
    augmentation_envelope: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50])

    # Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))
    phase8c_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet"))
    phase8c_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/manifests/phase8c_dataset_manifest.json"))

    # Experiment Output Directories
    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration"))
    configs_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/configs"))
    experiments_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/experiments"))
    metrics_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/metrics"))
    audits_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/audits"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/reports"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/figures"))

    def get_config_hash(self) -> str:
        cfg_dict = {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "baseline_corpus_version": self.baseline_corpus_version,
            "calibrated_corpus_version": self.calibrated_corpus_version,
            "candidates": self.candidates,
            "approved_horizons": self.approved_horizons,
            "dev_train_start_date": self.dev_train_start_date,
            "dev_train_end_date": self.dev_train_end_date,
        }
        encoded = json.dumps(cfg_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "baseline_corpus_version": self.baseline_corpus_version,
            "calibrated_corpus_version": self.calibrated_corpus_version,
            "config_sha256": self.get_config_hash(),
            "candidates": self.candidates,
            "approved_horizons": self.approved_horizons,
            "recommended_augmentation_ratio": self.recommended_augmentation_ratio,
            "augmentation_envelope": self.augmentation_envelope,
            "source_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
        }
