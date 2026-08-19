"""
AtmosIQ Phase 10B: Alerting Governance & Deterministic Rollback Rules Engine.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import time
import logging

logger = logging.getLogger(__name__)


class Phase10BAlertingEngine:
    """Evaluates operational monitoring telemetry and triggers tiered alerts and rollback rules."""

    def __init__(self, manifests_dir: Path):
        self.manifests_dir = Path(manifests_dir)

    def evaluate_telemetry(
        self,
        mae_current: float,
        mae_baseline: float,
        bias_current: float,
        psi_max: float,
        coverage_90: float,
        contract_violations_count: int,
        model_version: str,
        latency_ms: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Evaluates operational metrics and generates structured alerts."""
        alerts = []
        now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 1. Contract Violation Audit (RED)
        if contract_violations_count > 0:
            alerts.append({
                "alert_id": "ALT_CONTRACT_VIOLATION_CRITICAL",
                "timestamp_utc": now_utc,
                "severity": "RED",
                "metric": "contract_violations_count",
                "observed_value": contract_violations_count,
                "threshold": 0,
                "model_version": model_version,
                "recommended_action": "TRIGGER_IMMEDIATE_ROLLBACK_OR_SAFE_HALT",
                "description": "Inference input contract violation detected. Malformed payload must be safely rejected.",
            })

        # 2. Performance Degradation (YELLOW / ORANGE / RED)
        mae_ratio = mae_current / (mae_baseline + 1e-6)
        if mae_ratio > 1.50:
            alerts.append({
                "alert_id": "ALT_PERFORMANCE_DEGRADATION_CRITICAL",
                "timestamp_utc": now_utc,
                "severity": "RED",
                "metric": "mae_degradation_ratio",
                "observed_value": float(mae_ratio),
                "threshold": 1.50,
                "model_version": model_version,
                "recommended_action": "INITIATE_ROLLBACK_TO_PREVIOUS_VERSION",
                "description": f"MAE degraded by {(mae_ratio-1)*100:.1f}% relative to baseline.",
            })
        elif mae_ratio > 1.25:
            alerts.append({
                "alert_id": "ALT_PERFORMANCE_DEGRADATION_MATERIAL",
                "timestamp_utc": now_utc,
                "severity": "ORANGE",
                "metric": "mae_degradation_ratio",
                "observed_value": float(mae_ratio),
                "threshold": 1.25,
                "model_version": model_version,
                "recommended_action": "INVESTIGATE_ROOT_CAUSE_AND_ATMOSPHERIC_REGIME",
                "description": f"Material MAE increase of {(mae_ratio-1)*100:.1f}%.",
            })
        elif mae_ratio > 1.10:
            alerts.append({
                "alert_id": "ALT_PERFORMANCE_DEGRADATION_WARNING",
                "timestamp_utc": now_utc,
                "severity": "YELLOW",
                "metric": "mae_degradation_ratio",
                "observed_value": float(mae_ratio),
                "threshold": 1.10,
                "model_version": model_version,
                "recommended_action": "CONTINUE_MONITORING_INCREASE_SAMPLING",
                "description": "Moderate increase in forecast error.",
            })

        # 3. Calibration Bias Audit
        if abs(bias_current) > 15.0:
            alerts.append({
                "alert_id": "ALT_CALIBRATION_BIAS_CRITICAL",
                "timestamp_utc": now_utc,
                "severity": "ORANGE",
                "metric": "absolute_prediction_bias",
                "observed_value": float(abs(bias_current)),
                "threshold": 15.0,
                "model_version": model_version,
                "recommended_action": "FLAG_CALIBRATION_REVIEW_REQUIRED",
                "description": f"Prediction bias ({bias_current:.2f} µg/m³) exceeds tolerance.",
            })

        # 4. Uncertainty Undercoverage
        if coverage_90 < 0.80:
            alerts.append({
                "alert_id": "ALT_UNCERTAINTY_UNDERCOVERAGE_MATERIAL",
                "timestamp_utc": now_utc,
                "severity": "ORANGE",
                "metric": "empirical_90_coverage",
                "observed_value": float(coverage_90),
                "threshold": 0.80,
                "model_version": model_version,
                "recommended_action": "EXPAND_PREDICTION_INTERVALS_RECALIBRATE_CONFORMAL",
                "description": f"90% interval coverage dropped to {coverage_90*100:.1f}%.",
            })

        # 5. Feature Drift (PSI)
        if psi_max > 0.50:
            alerts.append({
                "alert_id": "ALT_FEATURE_DRIFT_CRITICAL",
                "timestamp_utc": now_utc,
                "severity": "RED",
                "metric": "max_feature_psi",
                "observed_value": float(psi_max),
                "threshold": 0.50,
                "model_version": model_version,
                "recommended_action": "AUDIT_UPSTREAM_SENSOR_TELEMETRY",
                "description": "Severe feature distribution shift detected (PSI > 0.50).",
            })
        elif psi_max > 0.25:
            alerts.append({
                "alert_id": "ALT_FEATURE_DRIFT_MATERIAL",
                "timestamp_utc": now_utc,
                "severity": "ORANGE",
                "metric": "max_feature_psi",
                "observed_value": float(psi_max),
                "threshold": 0.25,
                "model_version": model_version,
                "recommended_action": "INVESTIGATE_FEATURE_DISTRIBUTIONS",
                "description": "Material feature distribution shift detected (PSI > 0.25).",
            })
        elif psi_max > 0.10:
            alerts.append({
                "alert_id": "ALT_FEATURE_DRIFT_WARNING",
                "timestamp_utc": now_utc,
                "severity": "YELLOW",
                "metric": "max_feature_psi",
                "observed_value": float(psi_max),
                "threshold": 0.10,
                "model_version": model_version,
                "recommended_action": "CONTINUE_MONITORING_FEATURE_SHIFT",
                "description": "Potential feature distribution shift detected (PSI > 0.10).",
            })

        # 6. Latency SLA Audit
        if latency_ms > 10.0:
            alerts.append({
                "alert_id": "ALT_LATENCY_SLA_WARNING",
                "timestamp_utc": now_utc,
                "severity": "YELLOW",
                "metric": "single_sequence_latency_ms",
                "observed_value": float(latency_ms),
                "threshold": 10.0,
                "model_version": model_version,
                "recommended_action": "INVESTIGATE_HOST_COMPUTE_LOAD",
                "description": f"Latency ({latency_ms:.2f} ms) exceeds SLA threshold.",
            })

        return alerts

    def export_alert_and_rollback_policies(
        self,
        current_model_version: str,
        rollback_target_version: str = "MODEL_V3_PRODUCTION"
    ) -> Tuple[Path, Path]:
        """Serializes formal alert governance policy and deterministic rollback contract."""
        alert_policy = {
            "policy_name": "AtmosIQ_Phase10B_Alerting_Governance_Policy",
            "version": "1.0.0",
            "tier_definitions": {
                "GREEN": "Normal operation. Telemetry within standard tolerance.",
                "YELLOW": "Potential drift or minor performance degradation. Continue inference, increase monitoring.",
                "ORANGE": "Material degradation (e.g. MAE +25%, Bias > 15 µg/m³). Requires investigation; do not auto-retrain.",
                "RED": "Critical operational failure (e.g. Contract violation, MAE +50%, NaN/Inf output). Trigger deterministic rollback.",
            },
            "sla_thresholds": {
                "max_single_sequence_latency_ms": 10.0,
                "max_batch_latency_ms": 50.0,
                "max_acceptable_bias_ug_m3": 15.0,
                "min_90_conformal_coverage": 0.80,
            }
        }
        alert_path = self.manifests_dir / "phase10b_alert_policy.json"
        with open(alert_path, "w") as f:
            json.dump(alert_policy, f, indent=4)

        rollback_policy = {
            "policy_name": "AtmosIQ_Phase10B_Deterministic_Rollback_Policy",
            "version": "1.0.0",
            "current_production_model": current_model_version,
            "rollback_target_model": rollback_target_version,
            "rollback_triggers": [
                "REPEATED_INFERENCE_CONTRACT_VIOLATIONS",
                "CORRUPTED_OR_NAN_INF_OUTPUTS",
                "MAE_DEGRADATION_EXCEEDING_50_PERCENT",
                "PERSISTENT_UNRELIABLE_CALIBRATION_BIAS",
                "MODEL_WEIGHT_OR_HASH_INTEGRITY_FAILURE"
            ],
            "rollback_mechanism": "CRYPTOGRAPHIC_POINTER_REVERSION_TO_LAST_APPROVED_VERSION",
            "automatic_retraining_allowed": False,
            "governance_approval_required_for_repromotion": True,
        }
        rollback_path = self.manifests_dir / "phase10b_rollback_policy.json"
        with open(rollback_path, "w") as f:
            json.dump(rollback_policy, f, indent=4)

        return alert_path, rollback_path
