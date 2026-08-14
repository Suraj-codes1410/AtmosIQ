import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Common Scientific Disclaimer
SCIENTIFIC_DISCLAIMER = (
    "PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != "
    "CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION. "
    "AtmosIQ attributions and counterfactual sensitivities explain model feature responses. "
    "They do NOT constitute physical chemical-transport simulations or direct causal emission measurements."
)

PERSISTENCE_CAVEAT = (
    "PM2.5 persistence features (pm25_persistence group) represent model dependence on prior atmospheric "
    "pollution state history and accumulated source presence. It is NOT an independent physical emission source."
)


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    shap_loaded: bool = True
    validation_loaded: bool = True
    counterfactual_loaded: bool = True
    dataset_loaded: bool = True
    integrity_check: str = "PASS"


class ModelInfoResponse(BaseModel):
    model_type: str = "RandomForestRegressor"
    model_version: str = "phase3g_rf_v1"
    model_hash: str = "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"
    dataset_version: str = "v2"
    dataset_hash: str = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    feature_count: int = 147
    shap_available: bool = True
    counterfactual_available: bool = True
    attribution_groups: List[str] = [
        "pm25_persistence",
        "meteorology",
        "wind_ventilation",
        "biomass_burning",
        "calendar_seasonal"
    ]
    scientific_disclaimer: str = SCIENTIFIC_DISCLAIMER


class PredictionResponse(BaseModel):
    date: str
    observed_pm25: Optional[float] = None
    predicted_pm25: float
    persistence_baseline: Optional[float] = None
    prediction_error: Optional[float] = None
    model_version: str = "phase3g_rf_v1"
    dataset_version: str = "v2"
    pollution_category: str = "Normal"


class FeatureAttributionItem(BaseModel):
    feature_name: str
    shap_value: float
    attribution_group: str


class GroupAttributionItem(BaseModel):
    group_name: str
    signed_shap_sum: float
    abs_shap_sum: float
    share_pct: float


class AttributionResponse(BaseModel):
    date: str
    predicted_pm25: float
    base_value: float
    top_positive_features: List[FeatureAttributionItem]
    top_negative_features: List[FeatureAttributionItem]
    group_attributions: List[GroupAttributionItem]
    dominant_group: str
    scientific_disclaimer: str = SCIENTIFIC_DISCLAIMER
    persistence_caveat: str = PERSISTENCE_CAVEAT


class GroupValidationEvidence(BaseModel):
    group_name: str
    supporting_indicator: str
    relationship: str
    evidence_status: str
    observed_value: Optional[float] = None


class CounterEvidenceItem(BaseModel):
    group: str
    reason: str
    severity: str = "moderate"


class EnvironmentalValidationResponse(BaseModel):
    date: str
    validation_status: str = "PASS"
    group_evidence: List[GroupValidationEvidence]
    has_counter_evidence: bool = False
    counter_evidence_conflicts: List[CounterEvidenceItem] = []


class CounterfactualResponse(BaseModel):
    date: str
    scenario: str
    target_group: str
    baseline_prediction: float
    counterfactual_prediction: float
    delta_prediction: float
    ood_status: str = "PASS"
    plausibility: str = "PASS"
    shap_directional_consistency: bool = True
    confidence: str = "HIGH"
    interpretation: str
    scientific_disclaimer: str = SCIENTIFIC_DISCLAIMER


class ConfidenceResponse(BaseModel):
    date: str
    confidence_level: str  # HIGH, MODERATE, LOW, INVALID
    confidence_score: float
    supporting_reasons: List[str]
    risk_factors: List[str]


class ExtremeEventSummary(BaseModel):
    is_extreme_event: bool
    extreme_threshold: float = 306.81
    peak_pm25: float
    dominant_source_group: str
    event_severity: str


class EventResponse(BaseModel):
    event_id: str
    start_date: str
    end_date: str
    peak_date: str
    peak_pm25: float
    dominant_group: str
    duration_days: int
    group_attributions: Dict[str, float]
    biomass_cf_delta: float
    wind_cf_delta: float
    combined_cf_delta: float
    confidence_level: str
    seasonal_regime: str
    has_counter_evidence: bool = False


class DecisionSupportResponse(BaseModel):
    date: str
    prediction: PredictionResponse
    attribution: AttributionResponse
    validation: EnvironmentalValidationResponse
    counterfactual_scenarios: Dict[str, CounterfactualResponse]
    confidence: ConfidenceResponse
    extreme_event_analysis: Optional[ExtremeEventSummary] = None
    scientific_interpretation: str
    scientific_limitations: str = SCIENTIFIC_DISCLAIMER
