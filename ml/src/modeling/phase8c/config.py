"""
AtmosIQ Phase 8C: Production Release Configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class ReleaseConfigPhase8C:
    # Release Metadata
    phase_name: str = "Phase 8C"
    phase_version: str = "1.0.0"
    corpus_name: str = "AtmosIQ_Synthetic_Production"
    corpus_version: str = "v1.0.0"
    generator_version: str = "HP-STG-v1.0.0"

    # Mandatory Restrictions (Inherited from Phase 7C & 8B)
    approved_horizons: List[int] = field(default_factory=lambda: [14, 30])
    recommended_augmentation_ratio: float = 0.25
    allowed_augmentation_ratios: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50])
    controlled_upper_bound_ratio: float = 0.50
    prohibited_ratios: List[float] = field(default_factory=lambda: [1.00])

    # Extreme Tail Filter Thresholds (Restriction C)
    extreme_pm25_threshold: float = 250.0
    vi_threshold: float = 4500.0
    precipitation_threshold: float = 2.0

    # Partitions
    source_partition_name: str = "2020-2021"
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Directory Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))

    # Source Phase 8B Corpus
    phase8b_batches_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/batches"))
    phase8b_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8b/manifests/phase8b_manifest.json"))

    # Phase 8C Release Output Directory
    release_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release"))
    synthetic_dataset_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/synthetic_dataset"))
    manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/manifests"))
    audits_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/audits"))
    contracts_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/contracts"))
    hashes_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/hashes"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/reports"))

    def get_config_hash(self) -> str:
        cfg_dict = {
            "corpus_name": self.corpus_name,
            "corpus_version": self.corpus_version,
            "generator_version": self.generator_version,
            "approved_horizons": self.approved_horizons,
            "recommended_augmentation_ratio": self.recommended_augmentation_ratio,
            "allowed_augmentation_ratios": self.allowed_augmentation_ratios,
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
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "corpus_name": self.corpus_name,
            "corpus_version": self.corpus_version,
            "generator_version": self.generator_version,
            "config_sha256": self.get_config_hash(),
            "approved_horizons": self.approved_horizons,
            "recommended_augmentation_ratio": self.recommended_augmentation_ratio,
            "allowed_augmentation_ratios": self.allowed_augmentation_ratios,
            "controlled_upper_bound_ratio": self.controlled_upper_bound_ratio,
            "prohibited_ratios": self.prohibited_ratios,
            "source_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
        }
