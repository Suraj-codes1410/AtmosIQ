import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6f.config import DecisionSupportConfigPhase6F

logger = setup_logger("DecisionRulesPhase6F")


class DecisionRulesEnginePhase6F:
    """
    Deterministic Decision-Support Rule Engine for Phase 6F.
    Combines calibrated prediction intervals, OOD metrics, attribution stability,
    and counterfactual certainty into transparent reliability classifications.
    """

    def __init__(self, config: DecisionSupportConfigPhase6F):
        self.config = config

    def evaluate_decision_support(
        self,
        prediction_val: float,
        interval_data: Dict[str, Any],
        attribution_data: Dict[str, Any],
        ood_data: Dict[str, Any],
        counterfactual_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Applies deterministic rules to classify reliability and generate actionable recommendations.
        """
        flags = []
        rel_width = interval_data["interval_width"] / max(prediction_val, 10.0)
        regime = interval_data["pollution_regime"]
        ood_status = ood_data["ood_status"]

        # Check attribution stability among top dominant features
        dominant_feats = attribution_data["dominant_features"]
        feat_map = {f["feature_name"]: f for f in attribution_data["feature_contributions"]}
        dominant_stabilities = [feat_map[f]["sign_stability"] for f in dominant_feats if f in feat_map]
        avg_top_stab = float(sum(dominant_stabilities) / max(len(dominant_stabilities), 1))

        # 1. Uncertainty flags
        if regime == "Extreme":
            flags.append("EXTREME_REGIME_ELEVATED_SPREAD")
        if ood_status == "OOD":
            flags.append("OUT_OF_DISTRIBUTION_INPUT")
        elif ood_status == "NEAR_OOD":
            flags.append("MODERATE_DISTRIBUTION_SHIFT")
        if rel_width > self.config.relative_width_wide_threshold:
            flags.append("WIDE_PREDICTION_INTERVAL")
        if avg_top_stab < self.config.moderate_stability_threshold:
            flags.append("LOW_ATTRIBUTION_STABILITY")

        # 2. Tier Classification Logic
        # Rule 1: High Reliability
        if (
            rel_width <= self.config.relative_width_wide_threshold
            and ood_status == "IN_DISTRIBUTION"
            and avg_top_stab >= self.config.moderate_stability_threshold
            and regime != "Extreme"
        ):
            tier = "HIGH_RELIABILITY"
            heuristic_score = 85.0 + min(15.0, (1.0 - rel_width) * 15.0)
            summary_statement = (
                f"High model reliability. Calibrated 90% prediction interval is well-constrained "
                f"([ {interval_data['lower_bound']:.1f}, {interval_data['upper_bound']:.1f} ] µg/m³), "
                f"features reside within historical distribution bounds, and dominant attributions exhibit high directional consensus."
            )
        # Rule 2: High Uncertainty
        elif (
            ood_status == "OOD"
            or rel_width > 0.85
            or (regime == "Extreme" and rel_width > 0.60)
        ):
            tier = "HIGH_UNCERTAINTY"
            heuristic_score = max(25.0, 50.0 - (rel_width * 20.0))
            summary_statement = (
                f"High forecast uncertainty. The forecast operates under elevated atmospheric dispersion or distribution shift. "
                f"Decision-makers should interpret point estimates cautiously and plan against the 90% upper bound ({interval_data['upper_bound']:.1f} µg/m³)."
            )
        # Rule 3: Moderate Reliability (Default)
        else:
            tier = "MODERATE_RELIABILITY"
            heuristic_score = 70.0 - (rel_width * 10.0)
            summary_statement = (
                f"Moderate model reliability. The forecast provides a standard calibrated interval "
                f"([ {interval_data['lower_bound']:.1f}, {interval_data['upper_bound']:.1f} ] µg/m³). "
                f"Environmental attributions reflect known seasonal patterns with acceptable stability."
            )

        return {
            "reliability_tier": tier,
            "reliability_index_heuristic": round(float(heuristic_score), 1),
            "relative_interval_width": float(rel_width),
            "dominant_attribution_stability": float(avg_top_stab),
            "uncertainty_flags": flags,
            "decision_support_summary": summary_statement
        }
