"""
AtmosIQ Phase 7C: Configuration Dataclass for Validation & ML Utility.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class ValidationConfigPhase7C:
    # Phase & Metadata
    phase_name: str = "Phase 7C"
    phase_version: str = "1.0.0"
    random_seed: int = 42

    # Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))
    
    synthetic_parquet_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/synthetic/phase7b/synthetic_trajectories.parquet"))
    synthetic_csv_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/synthetic/phase7b/synthetic_trajectories.csv"))

    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase7c"))
    plot_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase7c/plots"))
    ml_utility_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase7c/ml_utility"))

    # Partitions
    dev_train_start: str = "2020-01-01"
    dev_train_end: str = "2021-12-31"
    locked_eval_start: str = "2022-01-01"
    locked_eval_end: str = "2024-12-31"

    # Predefined Acceptance Thresholds
    target_wasserstein_max: float = 0.15
    target_correlation_frobenius_max: float = 0.20
    target_acf_error_max: float = 0.08
    target_extreme_coherence_min: float = 0.95
    target_hard_constraint_compliance: float = 1.00

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "random_seed": self.random_seed,
            "dev_train_period": f"{self.dev_train_start} to {self.dev_train_end}",
            "locked_eval_period": f"{self.locked_eval_start} to {self.locked_eval_end}",
            "acceptance_targets": {
                "wasserstein_max": self.target_wasserstein_max,
                "correlation_frobenius_max": self.target_correlation_frobenius_max,
                "acf_error_max": self.target_acf_error_max,
                "extreme_coherence_min": self.target_extreme_coherence_min,
                "hard_constraint_compliance": self.target_hard_constraint_compliance,
            }
        }
