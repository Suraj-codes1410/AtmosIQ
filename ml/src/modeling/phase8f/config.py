"""
AtmosIQ Phase 8F: Configuration System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class Phase8FConfig:
    phase_name: str = "Phase 8F"
    phase_version: str = "1.0.0"
    canonical_production_corpus_version: str = "AtmosIQ_Synthetic_Production_v1.0.0"
    preferred_research_corpus_version: str = "AtmosIQ_Synthetic_Calibrated_v0.1.0"
    global_seed: int = 42

    # Partitions
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Upstream Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))
    
    phase8c_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet"))
    phase8c_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/manifests/phase8c_dataset_manifest.json"))
    phase8c_policy_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/manifests/synthetic_augmentation_policy.json"))

    phase8d_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet"))
    phase8d_config_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/configs/phase8d_calibration_config.json"))

    phase8e_contract_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/contracts/phase9_training_contract.json"))
    phase8e_ranking_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/rankings/corpus_candidate_ranking.csv"))

    # Output Directories
    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance"))
    audits_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/audits"))
    manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/manifests"))
    governance_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/governance"))
    hashes_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/hashes"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/reports"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/figures"))

    def get_config_hash(self) -> str:
        cfg = {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "canonical_production_corpus_version": self.canonical_production_corpus_version,
            "preferred_research_corpus_version": self.preferred_research_corpus_version,
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
            "dev_train_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
        }
