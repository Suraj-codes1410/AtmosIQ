"""
AtmosIQ Phase 8C: Phase 9 Deep Learning Training Contract Engine.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)


class Phase9TrainingContractEngine:
    """Generates the formal machine-readable training contract consumed by Phase 9."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def generate_contract(
        self,
        corpus_path: Path,
        corpus_sha256: str,
        output_path: Path
    ) -> Dict[str, Any]:
        contract_data = {
            "contract_name": "Phase 9 Deep Learning Training Contract",
            "contract_version": "1.0.0",
            "issuing_phase": "Phase 8C",
            "target_phase": "Phase 9 (Deep Learning Architectures)",
            "historical_development_partition": {
                "start_date": "2020-01-01",
                "end_date": "2021-12-31",
                "observation_count": 731,
                "dataset_source": "Dataset v3 (feature_dataset_frozen.csv)",
                "status": "APPROVED_FOR_TRAINING"
            },
            "synthetic_production_corpus": {
                "corpus_name": "AtmosIQ_Synthetic_Production",
                "corpus_version": "v1.0.0",
                "corpus_path": str(corpus_path),
                "corpus_sha256": corpus_sha256,
                "status": "APPROVED_FOR_AUGMENTATION"
            },
            "locked_real_evaluation_fold": {
                "start_date": "2022-01-01",
                "end_date": "2024-12-31",
                "observation_count": 1096,
                "dataset_source": "Dataset v3 (feature_dataset_frozen.csv)",
                "status": "LOCKED_EVALUATION_ONLY",
                "restriction": "FORBIDDEN from entering model training, loss computation, hyperparameter tuning, or generator calibration."
            },
            "augmentation_rules": {
                "default_recommended_ratio": 0.25,
                "allowed_experimental_envelope": [0.10, 0.25, 0.50],
                "hard_upper_limit": 0.50,
                "prohibited_ratio": 1.00
            },
            "feature_contract": {
                "required_feature_count": len(self.feature_registry),
                "feature_registry": self.feature_registry,
                "feature_names_source": "ml/models/production/v3/feature_registry.csv"
            },
            "governance_and_provenance_requirements": {
                "provenance_mandatory": True,
                "provenance_manifest": "manifests/synthetic_provenance_manifest.csv",
                "physical_laws_enforced": True,
                "extreme_tail_filtered": True,
                "hydrodynamic_identity_verified": True
            },
            "absolute_immutability_constraints": [
                "MODEL_V3_PRODUCTION remains frozen as benchmark baseline.",
                "ATMOSIQ_DECISION_SUPPORT v1.0.0 remains frozen as production decision-support stack.",
                "Phase 6F conformal uncertainty parameters remain immutable.",
                "Dataset v1, v2, v3 remain immutable."
            ]
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(contract_data, f, indent=4)

        logger.info(f"Phase 9 Training Contract written to {output_path}")
        return contract_data
