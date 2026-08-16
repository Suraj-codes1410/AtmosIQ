import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("EvidenceSynthesisPhase6F")


class EvidenceSynthesisPhase6F:
    """
    Model-Supported Environmental Evidence & Counter-Evidence Synthesizer for Phase 6F.
    Translates model features and SHAP attributions into verifiable atmospheric evidence statements.
    """

    @staticmethod
    def synthesize_evidence(
        feature_dict: Dict[str, float],
        attribution_result: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Synthesizes supporting factors and counter-evidence based on actual feature values and signed SHAP attributions.
        """
        supporting_factors = []
        counter_evidence = []

        feat_shap_map = {f["feature_name"]: f["shap_attribution"] for f in attribution_result["feature_contributions"]}

        # 1. PM2.5 Persistence / Inertia
        if "pm25_roll_mean_3d" in feature_dict:
            val = feature_dict["pm25_roll_mean_3d"]
            s = feat_shap_map.get("pm25_roll_mean_3d", 0.0)
            if val > 120.0 and s > 0:
                supporting_factors.append({
                    "factor": "High 3-Day PM2.5 Accumulation Baseline",
                    "feature": "pm25_roll_mean_3d",
                    "observed_value": f"{val:.1f} µg/m³",
                    "model_attribution": f"+{s:.2f} µg/m³",
                    "description": "Strong recent pollution memory elevates baseline forecast."
                })
            elif val < 60.0 and s < 0:
                counter_evidence.append({
                    "factor": "Low Recent PM2.5 Accumulation",
                    "feature": "pm25_roll_mean_3d",
                    "observed_value": f"{val:.1f} µg/m³",
                    "model_attribution": f"{s:.2f} µg/m³",
                    "description": "Clean prior air quality dampens next-day PM2.5 accumulation."
                })

        # 2. Ventilation & Inversion
        if "ventilation_index_1d" in feature_dict:
            val = feature_dict["ventilation_index_1d"]
            s = feat_shap_map.get("ventilation_index_1d", 0.0)
            if val < 2000.0 and s > 0:
                supporting_factors.append({
                    "factor": "Atmospheric Ventilation Deficit / Stagnation",
                    "feature": "ventilation_index_1d",
                    "observed_value": f"{val:.1f} m²/s",
                    "model_attribution": f"+{s:.2f} µg/m³",
                    "description": "Poor atmospheric ventilation restricts horizontal and vertical pollutant dispersion."
                })
            elif val >= 4000.0 and s < 0:
                counter_evidence.append({
                    "factor": "Strong Atmospheric Dispersion Capability",
                    "feature": "ventilation_index_1d",
                    "observed_value": f"{val:.1f} m²/s",
                    "model_attribution": f"{s:.2f} µg/m³",
                    "description": "High ventilation volume actively disperses particulates away from the surface layer."
                })

        # 3. Biomass Burning & Upwind Stubble Fires
        if "fire_hotspot_count_roll_mean_7d" in feature_dict:
            val = feature_dict["fire_hotspot_count_roll_mean_7d"]
            s = feat_shap_map.get("fire_hotspot_count_roll_mean_7d", 0.0)
            if val > 40.0 and s > 0:
                supporting_factors.append({
                    "factor": "Elevated Upwind Regional Stubble Fire Activity",
                    "feature": "fire_hotspot_count_roll_mean_7d",
                    "observed_value": f"{val:.1f} active hotspots",
                    "model_attribution": f"+{s:.2f} µg/m³",
                    "description": "Sustained regional biomass fire activity contributes positive model attribution."
                })
            elif val < 10.0:
                counter_evidence.append({
                    "factor": "Low Regional Open Fire Activity",
                    "feature": "fire_hotspot_count_roll_mean_7d",
                    "observed_value": f"{val:.1f} active hotspots",
                    "model_attribution": f"{s:.2f} µg/m³",
                    "description": "Minimal upwind agricultural burning input detected."
                })

        # 4. Precipitation & Aerosol Washout
        if "rain_event_1d" in feature_dict and feature_dict["rain_event_1d"] > 0:
            val = feature_dict["rain_event_1d"]
            s = feat_shap_map.get("rain_event_1d", 0.0)
            counter_evidence.append({
                "factor": "Precipitation Wet Deposition Washout",
                "feature": "rain_event_1d",
                "observed_value": "Active Rain Event",
                "model_attribution": f"{s:.2f} µg/m³",
                "description": "Rainfall aerosol scavenging removes airborne particulates."
            })

        return {
            "supporting_factors": supporting_factors,
            "counter_evidence": counter_evidence
        }
