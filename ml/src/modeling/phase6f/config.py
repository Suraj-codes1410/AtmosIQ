import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class DecisionSupportConfigPhase6F:
    project: str = "AtmosIQ"
    phase: str = "Phase 6F"
    experiment_name: str = "Uncertainty-Aware Decision Support, Final Integration & Production Acceptance"
    dataset_version: str = "Dataset_v3"
    dataset_sha256: str = "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736"
    production_model_name: str = "MODEL_V3_PRODUCTION"
    production_model_sha256: str = "9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210"
    production_uncertainty_method: str = "normalized_conformal"
    production_uncertainty_version: str = "1.0.0"
    decision_support_version: str = "1.0.0"
    target_variable: str = "pm25"
    feature_count: int = 35
    
    # Supported nominal coverage levels
    supported_coverage_levels: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95])
    default_nominal_coverage: float = 0.90

    # Decision rule thresholds
    relative_width_narrow_threshold: float = 0.35  # interval width / prediction < 35% -> narrow
    relative_width_wide_threshold: float = 0.65    # interval width / prediction > 65% -> wide
    ood_in_distribution_threshold: float = 2.0
    ood_near_ood_threshold: float = 3.5
    high_stability_threshold: float = 0.90
    moderate_stability_threshold: float = 0.70

    # Walk-forward folds (2022–2024, N=1,096)
    walk_forward_folds: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"fold": 1, "train_years": [2020, 2021], "eval_year": 2022},
        {"fold": 2, "train_years": [2020, 2021, 2022], "eval_year": 2023},
        {"fold": 3, "train_years": [2020, 2021, 2022, 2023], "eval_year": 2024}
    ])

    # Scientific disclaimer text
    scientific_disclaimer: str = (
        "PREDICTION INTERVAL != PHYSICAL ATMOSPHERIC UNCERTAINTY. "
        "SHAP ATTRIBUTION IS NOT CAUSAL ATTRIBUTION. "
        "COUNTERFACTUAL MODEL RESPONSE IS NOT A CAUSAL INTERVENTION EFFECT. "
        "All estimates represent statistical behavior of the learned predictive model under specified inputs."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, json_path: Path) -> "DecisionSupportConfigPhase6F":
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)
