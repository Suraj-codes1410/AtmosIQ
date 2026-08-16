"""
AtmosIQ Phase 7B: Configuration Dataclass for HP-STG & Physics Constraints.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class SyntheticConfigPhase7B:
    # Generator Version & Metadata
    generator_name: str = "HP-STG"
    generator_version: str = "1.0.0"
    data_origin: str = "synthetic"
    random_seed: int = 42

    # Paths
    root_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ"))
    dataset_v3_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/modeling/v3/feature_dataset_frozen.csv"))
    feature_registry_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/models/production/v3/feature_registry.csv"))
    freeze_manifest_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json"))
    phase7a_spec_path: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/docs/phase7/phase7a_physics_informed_synthetic_data_spec.md"))
    
    synthetic_data_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/data/synthetic/phase7b"))
    exp_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase7b"))
    plot_dir: Path = field(default_factory=lambda: Path("/home/suraj/atmosIQ/ml/experiments/phase7b/plots"))

    # Training Data Policy (2020-01-01 to 2021-12-31, N=731 days)
    training_start_date: str = "2020-01-01"
    training_end_date: str = "2021-12-31"
    locked_test_start_date: str = "2022-01-01"

    # Trajectory Generation Parameters
    trajectory_lengths: List[int] = field(default_factory=lambda: [14, 30, 90])
    num_trajectories_per_length: Dict[int, int] = field(default_factory=lambda: {14: 15, 30: 15, 90: 5})
    target_total_synthetic_days: int = 1110  # 15*14 + 15*30 + 5*90 = 210 + 450 + 450 = 1,110 days

    # Pollution Regimes (µg/m³)
    regime_low_threshold: float = 60.0
    regime_mod_threshold: float = 120.0
    regime_extreme_threshold: float = 250.0

    # Physics Constraints
    min_pm25: float = 0.0
    max_pm25: float = 500.0
    min_pblh: float = 150.0
    max_pblh: float = 3000.0
    min_wind_speed: float = 0.0
    max_wind_speed: float = 60.0
    min_rainfall: float = 0.0
    max_rainfall: float = 350.0
    min_humidity: float = 5.0
    max_humidity: float = 100.0
    min_temp: float = 0.0
    max_temp: float = 50.0

    # Target Acceptance Criteria for Phase 7B Pre-Check
    target_wasserstein_max: float = 0.15
    target_correlation_frobenius_max: float = 0.20
    target_acf_error_max: float = 0.08
    target_extreme_coherence_min: float = 0.95
    target_hard_constraint_compliance: float = 1.00

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "data_origin": self.data_origin,
            "random_seed": self.random_seed,
            "training_period": f"{self.training_start_date} to {self.training_end_date}",
            "locked_test_start_date": self.locked_test_start_date,
            "trajectory_lengths": self.trajectory_lengths,
            "num_trajectories_per_length": self.num_trajectories_per_length,
            "target_total_synthetic_days": self.target_total_synthetic_days,
            "regime_thresholds": {
                "low": self.regime_low_threshold,
                "moderate": self.regime_mod_threshold,
                "extreme": self.regime_extreme_threshold,
            },
            "acceptance_targets": {
                "wasserstein_max": self.target_wasserstein_max,
                "correlation_frobenius_max": self.target_correlation_frobenius_max,
                "acf_error_max": self.target_acf_error_max,
                "extreme_coherence_min": self.target_extreme_coherence_min,
                "hard_constraint_compliance": self.target_hard_constraint_compliance,
            }
        }
