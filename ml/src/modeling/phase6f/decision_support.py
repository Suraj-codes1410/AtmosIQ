import sys
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6f.config import DecisionSupportConfigPhase6F
from ml.src.modeling.phase6f.uncertainty_adapter import UncertaintyAdapterPhase6F
from ml.src.modeling.phase6f.attribution_adapter import AttributionAdapterPhase6F
from ml.src.modeling.phase6f.counterfactual_adapter import CounterfactualAdapterPhase6F
from ml.src.modeling.phase6f.ood_adapter import OODAdapterPhase6F
from ml.src.modeling.phase6f.evidence import EvidenceSynthesisPhase6F
from ml.src.modeling.phase6f.decision_rules import DecisionRulesEnginePhase6F

logger = setup_logger("DecisionSupportServicePhase6F")


class AtmosIQDecisionSupportService:
    """
    AtmosIQ Phase 6F Unified Uncertainty-Aware Decision Support Service.
    Integrates point forecasting, calibrated conformal prediction intervals,
    TreeSHAP attributions, counterfactual simulations, OOD analysis, and evidence synthesis.
    """

    def __init__(
        self,
        config: Optional[DecisionSupportConfigPhase6F] = None,
        model_path: str = "ml/models/production/v3/model.joblib",
        feature_registry_path: str = "ml/models/production/v3/feature_registry.csv"
    ):
        self.config = config or DecisionSupportConfigPhase6F()
        self.model = joblib.load(Path(model_path))
        
        df_feat = pd.read_csv(Path(feature_registry_path))
        self.features = list(df_feat["feature_name"].values)
        assert len(self.features) == 35, f"Expected 35 features, found {len(self.features)}"

        # Initialize core adapters
        self.uncertainty_adapter = UncertaintyAdapterPhase6F(self.config)
        self.attribution_adapter = AttributionAdapterPhase6F(self.config, model_path=model_path)
        self.counterfactual_adapter = CounterfactualAdapterPhase6F(self.config, model_path=model_path)
        self.ood_adapter = OODAdapterPhase6F(self.config)
        self.evidence_engine = EvidenceSynthesisPhase6F()
        self.rules_engine = DecisionRulesEnginePhase6F(self.config)

    def validate_input(self, input_features: Dict[str, Any]) -> np.ndarray:
        """
        Validates input feature map for completeness, correct data types, and physical bounds.
        """
        missing = [f for f in self.features if f not in input_features]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} required features: {missing[:5]}...")

        vec = np.zeros(len(self.features), dtype=np.float64)
        for i, feat in enumerate(self.features):
            val = input_features[feat]
            if val is None or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Invalid non-numeric value for feature '{feat}': {val}")
            vec[i] = float(val)

        return vec

    def predict_with_decision_support(
        self,
        input_features: Dict[str, Any],
        nominal_coverage: float = 0.90,
        scenario_name: str = "combined_all_favorable"
    ) -> Dict[str, Any]:
        """
        Executes unified uncertainty-aware decision-support pipeline for a single input record.
        """
        # 1. Input Validation & Vector Alignment
        x_vec = self.validate_input(input_features)
        X_mat = x_vec.reshape(1, -1)

        # 2. Point Prediction
        raw_pred = float(self.model.predict(X_mat)[0])
        pred_val = float(max(0.0, raw_pred))  # Non-negativity constraint

        # 3. Calibrated Prediction Interval
        interval_data = self.uncertainty_adapter.compute_prediction_interval(pred_val, nominal_coverage=nominal_coverage)

        # 4. TreeSHAP Attribution & Environmental Groups
        attribution_data = self.attribution_adapter.compute_attribution(x_vec, self.features)

        # 5. Out-of-Distribution Assessment
        ood_data = self.ood_adapter.evaluate_ood(x_vec, self.features)

        # 6. Counterfactual Simulation
        cf_data = self.counterfactual_adapter.simulate(x_vec, self.features, scenario_name=scenario_name)

        # 7. Evidence & Counter-Evidence Synthesis
        evidence_data = self.evidence_engine.synthesize_evidence(input_features, attribution_data)

        # 8. Deterministic Decision Rules
        decision_data = self.rules_engine.evaluate_decision_support(
            pred_val,
            interval_data,
            attribution_data,
            ood_data,
            counterfactual_data=cf_data
        )

        # 9. Unified Canonical Schema Construction
        return {
            "prediction": {
                "value": round(pred_val, 2),
                "unit": "µg/m³",
                "pollution_regime": interval_data["pollution_regime"]
            },
            "prediction_interval": {
                "lower_bound": round(interval_data["lower_bound"], 2),
                "upper_bound": round(interval_data["upper_bound"], 2),
                "nominal_coverage": interval_data["nominal_coverage"],
                "interval_width": round(interval_data["interval_width"], 2),
                "method": interval_data["method"],
                "method_version": interval_data["method_version"],
                "unit": "µg/m³"
            },
            "attribution": {
                "base_value": round(attribution_data["base_value"], 2),
                "dominant_features": attribution_data["dominant_features"],
                "dominant_groups": attribution_data["dominant_groups"],
                "group_contributions": attribution_data["group_contributions"],
                "top_feature_contributions": attribution_data["feature_contributions"][:5]
            },
            "counterfactual": {
                "scenario_name": cf_data["scenario_name"],
                "intervened_groups": cf_data["intervened_groups"],
                "baseline_prediction": round(cf_data["baseline_prediction"], 2),
                "counterfactual_prediction": round(cf_data["counterfactual_prediction"], 2),
                "estimated_delta_pm25": round(cf_data["delta_pm25"], 2),
                "direction": cf_data["direction"],
                "directional_stability": round(cf_data["directional_stability"], 3),
                "counterfactual_interval_80": [round(cf_data["counterfactual_interval_80"][0], 2), round(cf_data["counterfactual_interval_80"][1], 2)],
                "interpretation": "Model-estimated response under specified intervention scenario (not a physical causal claim)."
            },
            "ood_assessment": {
                "ood_score": round(ood_data["ood_score"], 2),
                "ood_status": ood_data["ood_status"],
                "max_feature_divergence": ood_data["most_deviated_feature"],
                "max_z_score": round(ood_data["max_z_score"], 2),
                "warning_message": ood_data["warning_message"]
            },
            "evidence": {
                "supporting_factors": evidence_data["supporting_factors"],
                "counter_evidence": evidence_data["counter_evidence"]
            },
            "decision_support": {
                "reliability_tier": decision_data["reliability_tier"],
                "reliability_index_heuristic": decision_data["reliability_index_heuristic"],
                "uncertainty_flags": decision_data["uncertainty_flags"],
                "recommendation_summary": decision_data["decision_support_summary"]
            },
            "provenance": {
                "model_name": self.config.production_model_name,
                "model_sha256": self.config.production_model_sha256,
                "uncertainty_method": self.config.production_uncertainty_method,
                "uncertainty_version": self.config.production_uncertainty_version,
                "decision_support_version": self.config.decision_support_version,
                "dataset_version": self.config.dataset_version,
                "dataset_sha256": self.config.dataset_sha256,
                "feature_count": self.config.feature_count,
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            },
            "scientific_disclaimer": self.config.scientific_disclaimer
        }
