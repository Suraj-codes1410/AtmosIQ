"""
AtmosIQ Phase 8B: Scaling Configuration System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class ScalingConfigPhase8B:
    # Metadata
    phase_name: str = "Phase 8B"
    phase_version: str = "1.0.0"
    generator_version: str = "HP-STG-v1.0.0"
    global_master_seed: int = 42

    # Mandatory Restrictions (Inherited from Phase 7C & 8A)
    approved_horizons: List[int] = field(default_factory=lambda: [14, 30])
    approved_augmentation_ratios: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50])
    default_augmentation_ratio: float = 0.25

    # Scaling Batch Schedule: (batch_id, trajectory_count, description)
    # Default research scaling schedule
    scaling_schedule: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"batch_id": "batch_0001", "target_trajectories": 100, "label": "100_trajs"},
        {"batch_id": "batch_0002", "target_trajectories": 250, "label": "250_trajs"},
        {"batch_id": "batch_0003", "target_trajectories": 500, "label": "500_trajs"},
        {"batch_id": "batch_0004", "target_trajectories": 1000, "label": "1k_trajs"},
        {"batch_id": "batch_0005", "target_trajectories": 2500, "label": "2.5k_trajs"},
    ])

    # Extreme Tail Filter Thresholds (Restriction C)
    extreme_filter_enabled: bool = True
    extreme_pm25_threshold: float = 250.0
    vi_threshold: float = 4500.0
    precipitation_threshold: float = 2.0

    # Data Isolation & Partitions
    source_partition_name: str = "2020-2021"
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Sharding
    max_trajectories_per_shard: int = 250
    output_format: str = "parquet"

    # Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))

    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b"))
    batches_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/batches"))
    manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/manifests"))
    validation_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/validation"))
    metrics_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/metrics"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/reports"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/figures"))
    checksums_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/checksums"))

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validates configuration parameters."""
        for b in self.scaling_schedule:
            if b["target_trajectories"] <= 0:
                raise ValueError(f"Invalid target trajectory count in batch {b['batch_id']}: {b['target_trajectories']}")

    def get_config_hash(self) -> str:
        """Computes deterministic SHA-256 hash of configuration parameters."""
        cfg_dict = {
            "generator_version": self.generator_version,
            "phase_version": self.phase_version,
            "global_master_seed": self.global_master_seed,
            "approved_horizons": self.approved_horizons,
            "approved_augmentation_ratios": self.approved_augmentation_ratios,
            "extreme_pm25_threshold": self.extreme_pm25_threshold,
            "vi_threshold": self.vi_threshold,
            "precipitation_threshold": self.precipitation_threshold,
            "dev_train_start_date": self.dev_train_start_date,
            "dev_train_end_date": self.dev_train_end_date,
            "scaling_schedule": self.scaling_schedule,
        }
        encoded = json.dumps(cfg_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "generator_version": self.generator_version,
            "global_master_seed": self.global_master_seed,
            "config_sha256": self.get_config_hash(),
            "approved_horizons": self.approved_horizons,
            "approved_augmentation_ratios": self.approved_augmentation_ratios,
            "extreme_filter_enabled": self.extreme_filter_enabled,
            "extreme_pm25_threshold": self.extreme_pm25_threshold,
            "vi_threshold": self.vi_threshold,
            "precipitation_threshold": self.precipitation_threshold,
            "source_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
            "scaling_schedule": self.scaling_schedule,
        }
