"""
AtmosIQ Phase 8E: Phase 9 Deep Learning Training Contract Manager.
"""

from pathlib import Path
from typing import Dict, Any, List
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class Phase9ContractManager:
    """Issues and updates the authoritative Phase 9 Deep Learning Training Contract."""

    def __init__(self, contracts_dir: Path):
        self.contracts_dir = Path(contracts_dir)
        self.contracts_dir.mkdir(parents=True, exist_ok=True)

    def generate_contract(
        self,
        preferred_corpus_name: str,
        preferred_corpus_version: str,
        preferred_corpus_sha256: str,
        recommended_augmentation: float = 0.25,
        max_augmentation: float = 0.50,
        admission_status: str = "APPROVED_WITH_RESTRICTIONS"
    ) -> Dict[str, Any]:
        contract_data = {
            "contract_name": "AtmosIQ_Phase9_Deep_Learning_Training_Contract",
            "contract_version": "v1.1.0",
            "admission_status": admission_status,
            "training_partitions": {
                "real_historical_development_train": {
                    "start_date": "2020-01-01",
                    "end_date": "2021-12-31",
                    "observations": 731,
                    "role": "PRIMARY_REAL_GROUND_TRUTH",
                },
                "locked_evaluation_benchmark": {
                    "start_date": "2022-01-01",
                    "end_date": "2024-12-31",
                    "observations": 1096,
                    "role": "STRICT_EVALUATION_ONLY_ZERO_TRAINING_LEAKAGE",
                },
            },
            "synthetic_training_corpus": {
                "preferred_corpus_name": preferred_corpus_name,
                "preferred_corpus_version": preferred_corpus_version,
                "preferred_corpus_sha256": preferred_corpus_sha256,
                "recommended_augmentation_ratio": recommended_augmentation,
                "maximum_approved_augmentation_ratio": max_augmentation,
                "prohibited_augmentation_ratio": 1.00,
            },
            "approved_sequence_windows": [7, 14, 30],
            "default_sequence_window": 14,
            "approved_architectures": [
                "LSTM",
                "Temporal_CNN_TCN",
                "Temporal_Transformer",
                "Informer_Temporal_Attention",
            ],
            "preprocessing_contract": {
                "normalization_fitting": "EXCLUSIVELY_ON_2020_2021_REAL_DATA",
                "synthetic_normalization": "APPLY_REAL_SCALER_PARAMETERS",
                "target_scaling": "IDENTITY_NON_NEGATIVE",
            },
            "scientific_disclaimers": [
                "SYNTHETIC DATA != OBSERVED DATA",
                "PHYSICS-INFORMED != PHYSICALLY EXACT",
                "STATISTICAL FIDELITY != CAUSAL VALIDATION",
                "ML UTILITY != SCIENTIFIC TRUTH",
                "SYNTHETIC AUGMENTATION != REAL-WORLD OBSERVATION",
            ],
        }

        # Calculate contract hash
        contract_hash = hashlib.sha256(json.dumps(contract_data, sort_keys=True).encode("utf-8")).hexdigest()
        contract_data["contract_sha256"] = contract_hash

        contract_file = self.contracts_dir / "phase9_training_contract.json"
        with open(contract_file, "w") as f:
            json.dump(contract_data, f, indent=4)

        logger.info(f"Phase 9 Training Contract successfully generated at {contract_file} (SHA: {contract_hash[:16]}...).")
        return contract_data
