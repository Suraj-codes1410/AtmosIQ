"""
AtmosIQ Phase 9C–9D: Configuration System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class Phase9CDConfig:
    phase_name: str = "Phase 9C–9D"
    phase_version: str = "1.0.0"
    canonical_production_corpus_version: str = "AtmosIQ_Synthetic_Production_v1.0.0"
    preferred_research_corpus_version: str = "AtmosIQ_Synthetic_Calibrated_v0.1.0"
    target_variable: str = "pm25"
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2025])
    default_seed: int = 42
    extreme_threshold: float = 250.0 # µg/m³
    prediction_interval_alpha: float = 0.10 # 90% prediction intervals (also computes 80% and 95%)

    # Partitions
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Augmentation Rules
    recommended_augmentation_ratio: float = 0.25
    controlled_upper_bound_ratio: float = 0.50
    prohibited_augmentation_ratio: float = 1.00

    # Sequence Configuration
    sequence_window: int = 14
    feature_dim: int = 35

    # Upstream Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))

    phase8c_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet"))
    phase8d_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet"))
    phase8e_contract_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/contracts/phase9_training_contract.json"))
    phase8f_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8f_governance/manifests/phase8f_artifact_manifest.json"))
    phase8g_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8g_integration/manifests/phase8g_integration_manifest.json"))
    phase8h_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8h_readiness/manifests/phase8h_training_manifest.json"))
    phase9_benchmarks_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9_deep_learning/benchmarks"))
    phase9_checkpoints_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9_deep_learning/checkpoints"))
    phase9ab_manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9ab_certification/manifests"))

    # Output Directories
    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9cd_hardening"))
    manifests_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9cd_hardening/manifests"))
    audits_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9cd_hardening/audits"))
    benchmarks_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9cd_hardening/benchmarks"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9cd_hardening/reports"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9cd_hardening/figures"))
    hashes_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase9cd_hardening/hashes"))

    # Candidate Designations
    research_candidate_version: str = "AtmosIQ_DL_TCN_CAL07_50_RESEARCH_v1.0.0"
    production_candidate_version: str = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0"
    fallback_production_candidate_version: str = "AtmosIQ_DL_LSTM_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0"

    def get_config_hash(self) -> str:
        cfg = {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "canonical_production_corpus_version": self.canonical_production_corpus_version,
            "preferred_research_corpus_version": self.preferred_research_corpus_version,
            "recommended_augmentation_ratio": self.recommended_augmentation_ratio,
            "controlled_upper_bound_ratio": self.controlled_upper_bound_ratio,
            "sequence_window": self.sequence_window,
            "feature_dim": self.feature_dim,
            "seeds": self.seeds,
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
            "sequence_window": self.sequence_window,
            "feature_dim": self.feature_dim,
            "seeds": self.seeds,
            "extreme_threshold": self.extreme_threshold,
            "dev_train_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
        }
