"""
AtmosIQ Phase 8G: Augmentation Policy Engine & Enforcement Gate.
"""

from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class AugmentationPolicyViolation(Exception):
    """Exception raised when an invalid or prohibited augmentation configuration is attempted."""
    pass


class Phase8GAugmentationPolicyEngine:
    """Enforces strict augmentation governance policy across dataset assembly."""

    def __init__(self, recommended_ratio: float = 0.25, max_ratio: float = 0.50):
        self.recommended_ratio = recommended_ratio
        self.max_ratio = max_ratio

    def validate_augmentation_request(
        self,
        ratio: float,
        is_stress_test: bool = False
    ) -> Dict[str, Any]:
        """Validates requested augmentation ratio against governance policy rules."""
        if ratio < 0.0:
            raise AugmentationPolicyViolation(f"Negative augmentation ratio {ratio} is physically invalid.")

        if ratio >= 1.0:
            raise AugmentationPolicyViolation(
                f"Attempted 100% synthetic training (ratio={ratio}). "
                "100% synthetic training is strictly PROHIBITED by AtmosIQ Governance Policy. "
                "Synthetic data is for augmentation only, not observational replacement."
            )

        if ratio > self.max_ratio:
            raise AugmentationPolicyViolation(
                f"Augmentation ratio {ratio:.2f} exceeds controlled upper bound of {self.max_ratio:.2f}."
            )

        if ratio > self.recommended_ratio and not is_stress_test:
            raise AugmentationPolicyViolation(
                f"Augmentation ratio {ratio:.2f} exceeds recommended production ratio ({self.recommended_ratio:.2f}) "
                "without explicit stress_test flag enabled."
            )

        if ratio == 0.0:
            tier = "REAL_HISTORICAL_ONLY"
            status = "APPROVED_BASELINE"
        elif ratio == 0.25:
            tier = "RECOMMENDED_PRODUCTION"
            status = "APPROVED_PRODUCTION_DEFAULT"
        elif ratio == 0.50:
            tier = "CONTROLLED_UPPER_BOUND"
            status = "APPROVED_STRESS_TEST_ONLY"
        else:
            tier = f"CONTROLLED_SUB_TARGET_{int(ratio*100)}PCT"
            status = "APPROVED_EXPERIMENTAL"

        return {
            "requested_ratio": ratio,
            "tier": tier,
            "status": status,
            "is_valid": True,
        }
