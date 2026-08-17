"""
AtmosIQ Phase 8C: Synthetic Augmentation Policy Engine.
"""

from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class SyntheticAugmentationPolicyEngine:
    """Manages and serializes the formal synthetic data augmentation policy for model training."""

    def __init__(self):
        self.policy_data = {
            "policy_name": "AtmosIQ Synthetic Augmentation Policy",
            "policy_version": "1.0.0",
            "source_phases": ["Phase 7C", "Phase 8B", "Phase 8C"],
            "recommended_ratio": 0.25,
            "recommended_ratio_label": "RECOMMENDED_PRODUCTION (25%)",
            "allowed_ratios": [0.10, 0.25, 0.50],
            "maximum_ratio": 0.50,
            "maximum_ratio_label": "CONTROLLED_UPPER_BOUND (50%)",
            "prohibited_ratios": [1.00],
            "prohibited_ratios_label": "NOT_RECOMMENDED (100% Augmentation / Synthetic Only)",
            "rationale": (
                "Empirical validation across Phase 7C and Phase 8B proved that Real Data + 25% Synthetic Data "
                "consistently delivers the optimal held-out generalization performance on the locked 2022-2024 fold "
                "(Test MAE 16.79 ug/m3 vs Real-Only 17.00 ug/m3, Extreme Event Error -1.42 ug/m3). "
                "A 50% augmentation envelope serves as a controlled stress-test limit, whereas 100% augmentation "
                "is strictly prohibited for production deployments due to distribution drift."
            ),
            "enforcement_rules": [
                "Downstream training pipelines must default to exactly 25% synthetic augmentation.",
                "Downstream pipelines must reject any request exceeding 50% synthetic augmentation.",
                "Synthetic samples must never replace real historical training observations.",
                "Synthetic data must strictly originate from AtmosIQ_Synthetic_Production_v1.0.0."
            ]
        }

    def generate_policy_file(self, output_path: Path) -> Dict[str, Any]:
        """Generates synthetic_augmentation_policy.json."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.policy_data, f, indent=4)
        logger.info(f"Synthetic augmentation policy written to {output_path}")
        return self.policy_data
