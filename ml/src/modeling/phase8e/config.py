"""
AtmosIQ Phase 8E: Configuration System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json


@dataclass
class Phase8EConfig:
    phase_name: str = "Phase 8E"
    phase_version: str = "1.0.0"
    baseline_corpus_version: str = "AtmosIQ_Synthetic_Production_v1.0.0"
    calibrated_corpus_version: str = "AtmosIQ_Synthetic_Calibrated_v0.1.0"
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2025])
    primary_seed: int = 42

    # Partitions
    dev_train_start_date: str = "2020-01-01"
    dev_train_end_date: str = "2021-12-31"
    locked_eval_start_date: str = "2022-01-01"
    locked_eval_end_date: str = "2024-12-31"

    # Temporal Windows
    approved_sequence_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    default_sequence_window: int = 14

    # Architectures
    architectures: List[str] = field(default_factory=lambda: ["LSTM", "TCN", "Transformer"])

    # Augmentation Configurations to evaluate
    configurations: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"id": "REAL_ONLY", "name": "Real Historical Only (2020-2021)", "corpus": "none", "ratio": 0.0},
        {"id": "REAL_PLUS_8C_10", "name": "Real + 10% Phase 8C Baseline", "corpus": "8C", "ratio": 0.10},
        {"id": "REAL_PLUS_8C_25", "name": "Real + 25% Phase 8C Baseline (Recommended)", "corpus": "8C", "ratio": 0.25},
        {"id": "REAL_PLUS_8C_50", "name": "Real + 50% Phase 8C Baseline (Upper Bound)", "corpus": "8C", "ratio": 0.50},
        {"id": "REAL_PLUS_8D_10", "name": "Real + 10% Phase 8D CAL-07", "corpus": "8D", "ratio": 0.10},
        {"id": "REAL_PLUS_8D_25", "name": "Real + 25% Phase 8D CAL-07 (Primary Test)", "corpus": "8D", "ratio": 0.25},
        {"id": "REAL_PLUS_8D_50", "name": "Real + 50% Phase 8D CAL-07 (Upper Bound)", "corpus": "8D", "ratio": 0.50},
    ])

    # Upstream Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))
    phase8c_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet"))
    phase8d_corpus_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet"))
    phase8d_config_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8d_calibration/configs/phase8d_calibration_config.json"))

    # Output Directories
    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness"))
    experiments_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/experiments"))
    audits_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/audits"))
    rankings_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/rankings"))
    contracts_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/contracts"))
    hashes_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/hashes"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/figures"))
    reports_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase8e_readiness/reports"))

    def get_config_hash(self) -> str:
        cfg = {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "baseline_corpus_version": self.baseline_corpus_version,
            "calibrated_corpus_version": self.calibrated_corpus_version,
            "seeds": self.seeds,
            "architectures": self.architectures,
            "approved_sequence_windows": self.approved_sequence_windows,
            "configurations": self.configurations,
            "dev_train_start_date": self.dev_train_start_date,
            "dev_train_end_date": self.dev_train_end_date,
        }
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "phase_version": self.phase_version,
            "config_sha256": self.get_config_hash(),
            "baseline_corpus_version": self.baseline_corpus_version,
            "calibrated_corpus_version": self.calibrated_corpus_version,
            "seeds": self.seeds,
            "architectures": self.architectures,
            "approved_sequence_windows": self.approved_sequence_windows,
            "configurations": self.configurations,
            "dev_train_partition": f"{self.dev_train_start_date} to {self.dev_train_end_date}",
            "locked_eval_partition": f"{self.locked_eval_start_date} to {self.locked_eval_end_date}",
        }
