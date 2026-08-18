"""
AtmosIQ Phase 8G: Configuration System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class Phase8GConfig:
    phase_name: str = "Phase 8G"
    phase_version: str = "1.0.0"
    canonical_production_corpus_version: str = "AtmosIQ_Synthetic_Production_v1.0.0"
    preferred_research_corpus_version: str = "AtmosIQ_Synthetic_Calibrated_v0.1.0"
    target_variable: str = "pm25"
    global_seed: int = 42

    # Partitions
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Augmentation Tiers
    recommended_augmentation_ratio: float = 0.25
    controlled_upper_bound_ratio: float = 0.50
    prohibited_augmentation_ratio: float = 1.00

    # Sequence Windows
    approved_sequence_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    default_sequence_window: int = 14

    # Architectures supported by the interface
    architectures: List[str] = field(default_factory=lambda: ["LSTM", "TCN", "Transformer"])

    # Upstream Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))

    phase8c_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet"))
    phase8d_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet"))
    phase8e_contract_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/contracts/phase9_training_contract.json"))
    phase8f_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/manifests/phase8f_artifact_manifest.json"))

    # Output Directories
    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration"))
    audits_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration/audits"))
    manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration/manifests"))
    interfaces_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration/interfaces"))
    hashes_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration/hashes"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration/reports"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration/figures"))

    def get_config_hash(self) -> str:
        cfg = {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "canonical_production_corpus_version": self.canonical_production_corpus_version,
            "preferred_research_corpus_version": self.preferred_research_corpus_version,
            "recommended_augmentation_ratio": self.recommended_augmentation_ratio,
            "controlled_upper_bound_ratio": self.controlled_upper_bound_ratio,
            "default_sequence_window": self.default_sequence_window,
            "dev_train_start_date": self.dev_train_start_date,
            "dev_train_end_date": self.dev_train_end_date,
            "locked_eval_start_date": self.locked_eval_start_date,
            "locked_eval_end_date": self.locked_eval_end_date,
        }
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "config_sha256": self.get_config_hash(),
            "canonical_production_corpus_version": self.canonical_production_corpus_version,
            "preferred_research_corpus_version": self.preferred_research_corpus_version,
            "recommended_augmentation_ratio": self.recommended_augmentation_ratio,
            "controlled_upper_bound_ratio": self.controlled_upper_bound_ratio,
            "prohibited_augmentation_ratio": self.prohibited_augmentation_ratio,
            "approved_sequence_windows": self.approved_sequence_windows,
            "default_sequence_window": self.default_sequence_window,
            "architectures": self.architectures,
            "dev_train_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
        }
