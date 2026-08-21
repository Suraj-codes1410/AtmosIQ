"""
AtmosIQ Phase 11B: Alert Policy Connectivity & Rollback Configuration Auditor.

Performs limited operational validation verifying that GREEN, YELLOW, ORANGE, and RED
alert levels map correctly to policy actions, and audits rollback target accessibility.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import logging

from .config import (
    CERTIFIED_RELEASE_ID,
    FALLBACK_TARGET,
)
from ml.src.modeling.phase10b.alerting import Phase10BAlertingEngine

logger = logging.getLogger(__name__)


class Phase11BAlertValidator:
    """Audits alert policy wiring and rollback configuration readability."""

    def __init__(self, manifests_dir: Path, observability_manifests_dir: Path):
        self.manifests_dir = Path(manifests_dir)
        self.obs_dir = Path(observability_manifests_dir)
        self.alerting_engine = Phase10BAlertingEngine(self.obs_dir)

    def validate_alert_policy_mappings(self) -> List[Dict[str, Any]]:
        """
        Tests 4 controlled operational scenarios to verify policy connectivity:
        1. Normal Operational Baseline -> GREEN
        2. Moderate Drift -> YELLOW
        3. Significant Performance Degradation -> ORANGE / RED
        4. Malformed Contract Violation -> RED
        """
        results = []

        # Scenario 1: Normal baseline telemetry
        alerts_green = self.alerting_engine.evaluate_telemetry(
            mae_current=33.62,
            mae_baseline=33.62,
            bias_current=-5.06,
            psi_max=0.05,
            coverage_90=91.5,
            contract_violations_count=0,
            model_version=CERTIFIED_RELEASE_ID,
            latency_ms=1.52,
        )
        has_red_orange = any(a["severity"] in ("RED", "ORANGE") for a in alerts_green)
        results.append({
            "scenario": "1. Normal Operational Baseline",
            "expected_severity": "GREEN (No Critical Alerts)",
            "observed_alert_count": len(alerts_green),
            "triggered_severities": [a["severity"] for a in alerts_green] if alerts_green else ["GREEN_NOMINAL"],
            "status": "PASS" if not has_red_orange else "FAIL",
            "operational_action": "NORMAL_PRODUCTION_SERVING",
        })

        # Scenario 2: Moderate performance warning (MAE degradation = 13%, YELLOW)
        alerts_yellow = self.alerting_engine.evaluate_telemetry(
            mae_current=38.0,
            mae_baseline=33.62,
            bias_current=-6.0,
            psi_max=0.20,
            coverage_90=89.0,
            contract_violations_count=0,
            model_version=CERTIFIED_RELEASE_ID,
            latency_ms=1.8,
        )
        has_warning = any(a["severity"] == "YELLOW" for a in alerts_yellow)
        results.append({
            "scenario": "2. Moderate Performance Warning (MAE=38.0)",
            "expected_severity": "YELLOW (Warning)",
            "observed_alert_count": len(alerts_yellow),
            "triggered_severities": [a["severity"] for a in alerts_yellow],
            "status": "PASS" if has_warning else "FAIL",
            "operational_action": alerts_yellow[0]["recommended_action"] if alerts_yellow else "LOG_AND_MONITOR",
        })

        # Scenario 3: Significant degradation (MAE degradation > 50%)
        alerts_red = self.alerting_engine.evaluate_telemetry(
            mae_current=55.0,
            mae_baseline=33.62,
            bias_current=-18.0,
            psi_max=0.45,
            coverage_90=78.0,
            contract_violations_count=0,
            model_version=CERTIFIED_RELEASE_ID,
            latency_ms=2.5,
        )
        has_red = any(a["severity"] == "RED" for a in alerts_red)
        results.append({
            "scenario": "3. Severe Performance Degradation (MAE=55.0)",
            "expected_severity": "RED (Critical Alert)",
            "observed_alert_count": len(alerts_red),
            "triggered_severities": [a["severity"] for a in alerts_red],
            "status": "PASS" if has_red else "FAIL",
            "operational_action": alerts_red[0]["recommended_action"] if alerts_red else "TRIGGER_ROLLBACK",
        })

        # Scenario 4: Malformed input contract violation
        alerts_violation = self.alerting_engine.evaluate_telemetry(
            mae_current=33.62,
            mae_baseline=33.62,
            bias_current=-5.06,
            psi_max=0.05,
            coverage_90=91.5,
            contract_violations_count=3,
            model_version=CERTIFIED_RELEASE_ID,
            latency_ms=1.52,
        )
        has_contract_red = any(a["severity"] == "RED" and "CONTRACT" in a.get("alert_id", "") for a in alerts_violation)
        results.append({
            "scenario": "4. Contract Input Violation (N=3)",
            "expected_severity": "RED (Immediate Rejection/Halt)",
            "observed_alert_count": len(alerts_violation),
            "triggered_severities": [a["severity"] for a in alerts_violation],
            "status": "PASS" if has_contract_red else "FAIL",
            "operational_action": alerts_violation[0]["recommended_action"] if alerts_violation else "SAFE_REJECTION",
        })

        return results

    def verify_rollback_configuration(self) -> Dict[str, Any]:
        """Verifies rollback policy readability and fallback model target accessibility."""
        rollback_policy_path = self.obs_dir / "phase10b_rollback_policy.json"
        model_registry_path  = self.obs_dir / "phase10b_model_registry.json"

        results = {
            "rollback_policy_accessible": False,
            "model_registry_accessible": False,
            "fallback_target_identified": False,
            "fallback_target_name": FALLBACK_TARGET,
            "status": "FAIL",
        }

        if rollback_policy_path.exists():
            with open(rollback_policy_path) as f:
                rb_data = json.load(f)
            results["rollback_policy_accessible"] = True
            results["rollback_triggers"] = rb_data.get("triggers", [])
            results["rollback_target"]   = rb_data.get("target_version", FALLBACK_TARGET)
            results["fallback_target_identified"] = (FALLBACK_TARGET in json.dumps(rb_data))

        if model_registry_path.exists():
            with open(model_registry_path) as f:
                _ = json.load(f)
            results["model_registry_accessible"] = True

        all_ok = results["rollback_policy_accessible"] and results["model_registry_accessible"]
        results["status"] = "PASS" if all_ok else "FAIL"
        return results
