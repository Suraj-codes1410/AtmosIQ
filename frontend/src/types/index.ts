export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  shap_loaded: boolean;
  validation_loaded: boolean;
  counterfactual_loaded: boolean;
  dataset_loaded: boolean;
  integrity_check: string;
}

export interface ModelInfoResponse {
  model_type: string;
  model_version: string;
  model_hash: string;
  dataset_version: string;
  dataset_hash: string;
  feature_count: number;
  shap_available: boolean;
  counterfactual_available: boolean;
  attribution_groups: string[];
  scientific_disclaimer: string;
}

export interface PredictionResponse {
  date: string;
  observed_pm25: number | null;
  predicted_pm25: number;
  persistence_baseline: number | null;
  prediction_error: number | null;
  model_version: string;
  dataset_version: string;
  pollution_category: string;
}

export interface FeatureAttributionItem {
  feature_name: string;
  shap_value: number;
  attribution_group: string;
}

export interface GroupAttributionItem {
  group_name: string;
  signed_shap_sum: number;
  abs_shap_sum: number;
  share_pct: number;
}

export interface AttributionResponse {
  date: string;
  predicted_pm25: number;
  base_value: number;
  top_positive_features: FeatureAttributionItem[];
  top_negative_features: FeatureAttributionItem[];
  group_attributions: GroupAttributionItem[];
  dominant_group: string;
  scientific_disclaimer: string;
  persistence_caveat: string;
}

export interface GroupValidationEvidence {
  group_name: string;
  supporting_indicator: string;
  relationship: string;
  evidence_status: string;
  observed_value: number | null;
}

export interface CounterEvidenceItem {
  group: string;
  reason: string;
  severity: string;
}

export interface EnvironmentalValidationResponse {
  date: string;
  validation_status: string;
  group_evidence: GroupValidationEvidence[];
  has_counter_evidence: boolean;
  counter_evidence_conflicts: CounterEvidenceItem[];
}

export interface CounterfactualResponse {
  date: string;
  scenario: string;
  target_group: string;
  baseline_prediction: number;
  counterfactual_prediction: number;
  delta_prediction: number;
  ood_status: string;
  plausibility: string;
  shap_directional_consistency: boolean;
  confidence: string;
  interpretation: string;
  scientific_disclaimer: string;
}

export interface ConfidenceResponse {
  date: string;
  confidence_level: 'HIGH' | 'MODERATE' | 'LOW' | 'INVALID';
  confidence_score: number;
  supporting_reasons: string[];
  risk_factors: string[];
}

export interface ExtremeEventSummary {
  is_extreme_event: boolean;
  extreme_threshold: number;
  peak_pm25: number;
  dominant_source_group: string;
  event_severity: string;
}

export interface EventResponse {
  event_id: string;
  start_date: string;
  end_date: string;
  peak_date: string;
  peak_pm25: number;
  dominant_group: string;
  duration_days: number;
  group_attributions: Record<string, number>;
  biomass_cf_delta: number;
  wind_cf_delta: number;
  combined_cf_delta: number;
  confidence_level: string;
  seasonal_regime: string;
  has_counter_evidence: boolean;
}

export interface EventCatalogItem {
  event_id: string;
  event_start: string;
  event_end: string;
  duration_days: number;
  peak_date: string;
  peak_pm25: number;
  mean_pm25: number;
  dominant_attribution_group: string;
  biomass_burning_shap?: number;
  meteorology_shap?: number;
  wind_ventilation_shap?: number;
  pm25_persistence_shap?: number;
}

export interface DecisionSupportResponse {
  date: string;
  prediction: PredictionResponse;
  attribution: AttributionResponse;
  validation: EnvironmentalValidationResponse;
  counterfactual_scenarios: Record<string, CounterfactualResponse>;
  confidence: ConfidenceResponse;
  extreme_event_analysis: ExtremeEventSummary | null;
  scientific_interpretation: string;
  scientific_limitations: string;
}
