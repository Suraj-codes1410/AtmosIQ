import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class ValidationConfigPhase6D:
    project: str = "AtmosIQ"
    phase: str = "Phase 6D"
    experiment_name: str = "Final Prediction Interval Validation & Production Selection"
    candidate_method: str = "normalized_conformal"
    dataset_version: str = "Dataset_v3"
    dataset_sha256: str = "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736"
    production_model_name: str = "MODEL_V3_PRODUCTION"
    production_model_sha256: str = "9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210"
    target_variable: str = "pm25"
    feature_count: int = 35
    
    # Nominal coverage levels
    nominal_coverage_levels: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95])
    
    # Walk-forward folds
    walk_forward_folds: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"fold": 1, "train_years": [2020, 2021], "eval_year": 2022},
        {"fold": 2, "train_years": [2020, 2021, 2022], "eval_year": 2023},
        {"fold": 3, "train_years": [2020, 2021, 2022, 2023], "eval_year": 2024}
    ])
    
    # Base model hyperparameters
    rf_base_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 9,
        "min_samples_split": 4,
        "min_samples_leaf": 5,
        "max_features": 0.7
    })
    
    # Extreme pollution thresholds for stress testing
    extreme_stress_thresholds: List[float] = field(default_factory=lambda: [100.0, 150.0, 200.0, 250.0, 300.0, 400.0])
    
    # Rolling stability window sizes (days)
    rolling_window_days: List[int] = field(default_factory=lambda: [30, 60, 90])
    
    # Pollution regimes (µg/m³) - Default Config A
    pollution_regimes_config_a: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Low": {"min": 0.0, "max": 60.0},
        "Moderate": {"min": 60.0, "max": 120.0},
        "High": {"min": 120.0, "max": 250.0},
        "Extreme": {"min": 250.0, "max": 1000.0}
    })

    # Alternative Config B
    pollution_regimes_config_b: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Low": {"min": 0.0, "max": 50.0},
        "Moderate": {"min": 50.0, "max": 100.0},
        "High": {"min": 100.0, "max": 200.0},
        "Extreme": {"min": 200.0, "max": 1000.0}
    })
    
    # Seasons
    seasons: Dict[str, List[int]] = field(default_factory=lambda: {
        "Winter": [12, 1, 2],
        "Summer": [3, 4, 5],
        "Monsoon": [6, 7, 8, 9],
        "Post-Monsoon": [10, 11]
    })
    
    epsilon: float = 1e-4
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, json_path: Path) -> "ValidationConfigPhase6D":
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)
