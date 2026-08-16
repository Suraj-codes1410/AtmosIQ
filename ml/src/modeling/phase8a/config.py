"""
AtmosIQ Phase 8A: Generation Configuration System with Strict Validation Constraints.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class GenerationConfigPhase8A:
    # Generator & Phase metadata
    generator_version: str = "HP-STG-v1.0.0"
    phase_version: str = "Phase 8A v1.0.0"
    dataset_version: str = "AtmosIQ-SYNTH-v8A"
    mode: str = "PILOT"  # "PILOT" or "SCALE"
    global_seed: int = 42

    # Mandatory Restrictions (Phase 7C Findings)
    approved_horizons: List[int] = field(default_factory=lambda: [14, 30])
    approved_augmentation_ratios: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50])
    default_augmentation_ratio: float = 0.25

    # Requested Generation Parameters
    trajectory_lengths: List[int] = field(default_factory=lambda: [14, 30])
    augmentation_ratio: float = 0.25
    pilot_trajectory_count: int = 6  # 3x 14-day, 3x 30-day (132 days)
    scale_trajectory_count: int = 24 # 12x 14-day, 12x 30-day (528 days)

    # Extreme Tail Filter Thresholds
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
    max_trajectories_per_shard: int = 10
    output_format: str = "parquet"

    # Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))

    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8a"))
    shards_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8a/shards"))
    manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8a/manifests"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8a/reports"))
    checksums_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8a/checksums"))
    data_synthetic_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/synthetic/phase8a"))

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validates configuration against mandatory Phase 8 restrictions."""
        # 1. Validate trajectory horizons
        for horizon in self.trajectory_lengths:
            if horizon not in self.approved_horizons:
                raise ValueError(
                    f"Unsupported trajectory horizon: {horizon} days. "
                    f"Phase 8 strictly restricts generation to approved horizons: {self.approved_horizons}."
                )

        # 2. Validate augmentation ratio
        if self.augmentation_ratio not in self.approved_augmentation_ratios:
            raise ValueError(
                f"Unsupported augmentation ratio: {self.augmentation_ratio}. "
                f"Phase 8 strictly restricts augmentation to approved ratios: {self.approved_augmentation_ratios}. "
                f"100% or unbounded augmentation is not an approved production configuration."
            )

        # 3. Validate mode
        if self.mode not in ["PILOT", "SCALE"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'PILOT' or 'SCALE'.")

    def get_config_hash(self) -> str:
        """Returns deterministic SHA-256 hash of configuration parameters."""
        cfg_dict = {
            "generator_version": self.generator_version,
            "phase_version": self.phase_version,
            "dataset_version": self.dataset_version,
            "mode": self.mode,
            "global_seed": self.global_seed,
            "trajectory_lengths": sorted(self.trajectory_lengths),
            "augmentation_ratio": self.augmentation_ratio,
            "extreme_filter_enabled": self.extreme_filter_enabled,
            "extreme_pm25_threshold": self.extreme_pm25_threshold,
            "vi_threshold": self.vi_threshold,
            "precipitation_threshold": self.precipitation_threshold,
            "dev_train_start_date": self.dev_train_start_date,
            "dev_train_end_date": self.dev_train_end_date,
        }
        encoded = json.dumps(cfg_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_version": self.generator_version,
            "phase_version": self.phase_version,
            "dataset_version": self.dataset_version,
            "mode": self.mode,
            "global_seed": self.global_seed,
            "config_sha256": self.get_config_hash(),
            "trajectory_lengths": self.trajectory_lengths,
            "augmentation_ratio": self.augmentation_ratio,
            "extreme_filter_enabled": self.extreme_filter_enabled,
            "extreme_pm25_threshold": self.extreme_pm25_threshold,
            "vi_threshold": self.vi_threshold,
            "precipitation_threshold": self.precipitation_threshold,
            "source_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
        }
