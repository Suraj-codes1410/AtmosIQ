"""
AtmosIQ Phase 11B: Production Monitoring Baseline & Limited Operational Validation Tests.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.src.modeling.phase11b.config import (
    CERTIFIED_RELEASE_ID,
    CERTIFIED_MODEL_SHA256,
    CERTIFIED_ARCHITECTURE,
    CERTIFIED_PARAMS,
    CERTIFIED_WINDOW,
    CERTIFIED_FEATURE_DIM,
    CERTIFIED_AUGMENTATION,
    CERTIFIED_PROTECTED_COUNT,
    FALLBACK_TARGET,
    PRODUCTION_FEATURES,
    SLA_SINGLE_INFERENCE_MS,
    SLA_BATCH_PIPELINE_MS,
    SLA_MAX_MEMORY_MB,
)
from ml.src.modeling.phase11b.provenance import Phase11BProvenanceAuditor
from ml.src.modeling.phase11b.latency import Phase11BLatencyReconciler
from ml.src.modeling.phase11b.baseline import Phase11BBaselineEngine
from ml.src.modeling.phase11b.alerts import Phase11BAlertValidator
from ml.src.modeling.phase11b.monitoring import Phase11BMonitoringEngine

ROOT = Path(__file__).parent.parent.parent
BUNDLE_DIR = ROOT / "ml/experiments/phase10d_release/release_bundle"
DATASET_PATH = ROOT / "ml/data/modeling/v3/feature_dataset_frozen.csv"
OBS_DIR = ROOT / "ml/experiments/phase10b_observability/manifests"
MANIFESTS_DIR = ROOT / "ml/experiments/phase11b_monitoring/manifests"


@pytest.fixture(scope="module")
def provenance_auditor():
    return Phase11BProvenanceAuditor(ROOT)


@pytest.fixture(scope="module")
def latency_reconciler():
    return Phase11BLatencyReconciler(BUNDLE_DIR)


@pytest.fixture(scope="module")
def baseline_engine():
    return Phase11BBaselineEngine(DATASET_PATH)


@pytest.fixture(scope="module")
def alert_validator():
    return Phase11BAlertValidator(MANIFESTS_DIR, OBS_DIR)


@pytest.fixture(scope="module")
def monitoring_engine():
    return Phase11BMonitoringEngine(BUNDLE_DIR, DATASET_PATH)


# ── 1. Immutability & Provenance Tests ────────────────────────────────────────

class TestPhase11BProvenance:
    def test_model_checkpoint_sha_match(self, provenance_auditor):
        is_match, actual_sha = provenance_auditor.verify_release_checkpoint_sha()
        assert is_match, f"Model SHA mismatch: {actual_sha} != {CERTIFIED_MODEL_SHA256}"

    def test_protected_artifacts_audit(self, provenance_auditor):
        all_passed, total, drift, details = provenance_auditor.audit_protected_artifacts()
        assert all_passed, f"Protected artifact drift detected (drift={drift})"
        assert total == CERTIFIED_PROTECTED_COUNT
        assert drift == 0


# ── 2. Latency Benchmarking & Reconciliation Tests ─────────────────────────────

class TestPhase11BLatencyReconciliation:
    def test_multi_layer_benchmark(self, latency_reconciler):
        bench = latency_reconciler.benchmark_multi_layer_latency(repetitions=20)
        rec = bench["phase11b_reconciliation"]

        # Raw model pass should be ~0.14 ms
        assert rec["raw_model_forward_mean_ms"] < 2.0
        # Full service API should be under SLA (< 10 ms)
        assert rec["full_service_api_mean_ms"] < SLA_SINGLE_INFERENCE_MS
        # Batch pipeline under SLA (< 50 ms)
        assert rec["batch_service_api_mean_ms"] < SLA_BATCH_PIPELINE_MS
        # Memory under SLA (< 256 MB)
        assert rec["peak_memory_mb"] < SLA_MAX_MEMORY_MB
        assert rec["sla_single_pass"] is True
        assert rec["sla_batch_pass"] is True
        assert rec["is_model_regression"] is False


# ── 3. Operational Baseline & Distribution Tests ───────────────────────────────

class TestPhase11BBaseline:
    def test_input_quality_audit(self, baseline_engine):
        df_iq = baseline_engine.audit_input_quality()
        assert len(df_iq) == CERTIFIED_FEATURE_DIM
        clean_count = sum(df_iq["input_quality_status"] == "PASS_CLEAN")
        assert clean_count == CERTIFIED_FEATURE_DIM

    def test_feature_monitoring_psi(self, baseline_engine):
        df_feat = baseline_engine.compute_feature_monitoring()
        assert len(df_feat) > 0
        assert "psi" in df_feat.columns
        assert "drift_severity" in df_feat.columns
        # None should be critical red drift
        red_count = sum(df_feat["drift_severity"].str.contains("RED", na=False))
        assert red_count == 0

    def test_prediction_monitoring_calculation(self, baseline_engine):
        preds_base = np.random.normal(95.0, 30.0, 500)
        preds_rep  = np.random.normal(98.0, 32.0, 500)
        summary = baseline_engine.compute_prediction_monitoring(preds_base, preds_rep)
        assert "prediction_psi" in summary
        assert "prediction_wasserstein_distance" in summary
        assert summary["prediction_psi"] >= 0.0

    def test_calibration_uncertainty_monitoring(self, baseline_engine):
        y_true = np.array([80.0, 120.0, 60.0, 200.0, 150.0])
        y_pred = np.array([78.0, 115.0, 65.0, 195.0, 145.0])
        res = baseline_engine.compute_calibration_uncertainty_monitoring(y_true, y_pred)
        assert res["mae_pm25"] > 0
        assert res["empirical_coverage_pct"] >= 80.0


# ── 4. Alert Policy & Rollback Verification Tests ─────────────────────────────

class TestPhase11BAlertsAndRollback:
    def test_alert_policy_mappings(self, alert_validator):
        results = alert_validator.validate_alert_policy_mappings()
        assert len(results) == 4
        # All 4 scenarios must pass
        for r in results:
            assert r["status"] == "PASS", f"Scenario failed: {r}"

    def test_rollback_target_verification(self, alert_validator):
        rb = alert_validator.verify_rollback_configuration()
        assert rb["status"] == "PASS"
        assert rb["rollback_policy_accessible"] is True
        assert rb["model_registry_accessible"] is True
        assert rb["fallback_target_name"] == FALLBACK_TARGET


# ── 5. Operational Stream Monitoring Tests ────────────────────────────────────

class TestPhase11BMonitoringStream:
    def test_stream_processing_sample(self, monitoring_engine):
        # Sample step 50 for quick test execution
        stream_res = monitoring_engine.run_operational_stream_monitoring(sample_step=50)
        summ = stream_res["summary"]
        assert summ["successful_inferences"] > 0
        assert summ["rejected_requests"] == 0
        assert summ["latency_mean_ms"] < SLA_SINGLE_INFERENCE_MS
        assert summ["sla_single_inference_pass"] is True
        assert summ["empirical_90_coverage_pct"] > 70.0
