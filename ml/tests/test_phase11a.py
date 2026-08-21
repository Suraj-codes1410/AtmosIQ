"""
AtmosIQ Phase 11A: Post-Release Smoke Validation Tests.

Focused smoke tests confirming that the certified v1.0.0 release
remains operationally reproducible. Does not retrain, recalibrate,
or modify any production artifact.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.src.modeling.phase11a.smoke import (
    Phase11ASmokeValidator,
    CERTIFIED_RELEASE_ID,
    CERTIFIED_MODEL_SHA256,
    CERTIFIED_ARCHITECTURE,
    CERTIFIED_PARAMS,
    CERTIFIED_WINDOW,
    CERTIFIED_FEATURE_DIM,
    CERTIFIED_PROTECTED_COUNT,
    DETERMINISM_TOLERANCE,
    PRODUCTION_FEATURES,
)
from ml.src.modeling.phase10d.deployment import Phase10DDeploymentService, ServiceContractException

ROOT = Path(__file__).parent.parent.parent
BUNDLE_DIR = ROOT / "ml/experiments/phase10d_release/release_bundle"


@pytest.fixture(scope="module")
def validator():
    return Phase11ASmokeValidator(ROOT)


@pytest.fixture(scope="module")
def service():
    return Phase10DDeploymentService(BUNDLE_DIR)


@pytest.fixture(scope="module")
def known_good_fixture():
    """Deterministic W=14 x D=35 fixture using certified feature names."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((CERTIFIED_WINDOW, CERTIFIED_FEATURE_DIM)).astype(np.float32)
    return pd.DataFrame(data, columns=PRODUCTION_FEATURES)


# ── Release identity ─────────────────────────────────────────────────────────

class TestReleaseIdentity:
    def test_release_manifest_exists(self):
        path = ROOT / "ml/experiments/phase10d_release/manifests/phase10d_release_manifest.json"
        assert path.exists(), "Phase 10D release manifest missing"

    def test_release_id_matches(self, validator):
        result = validator.check_release_identity()
        assert result["status"] == "PASS", f"Release ID mismatch: {result}"
        assert result["release_id"] == CERTIFIED_RELEASE_ID

    def test_release_md_exists(self):
        assert (ROOT / "release/v1.0.0/RELEASE.md").exists()

    def test_reconciliation_doc_exists(self):
        assert (ROOT / "docs/releases/ARTIFACT_COUNT_RECONCILIATION.md").exists()


# ── Model SHA-256 ─────────────────────────────────────────────────────────────

class TestModelSHA:
    def test_model_checkpoint_exists(self):
        assert (BUNDLE_DIR / "model_checkpoint.json").exists()

    def test_model_sha_matches(self, validator):
        result = validator.check_model_sha()
        assert result["status"] == "PASS", (
            f"Model SHA mismatch.\n"
            f"  Expected: {CERTIFIED_MODEL_SHA256}\n"
            f"  Actual:   {result.get('actual_sha')}"
        )


# ── Protected artifact integrity ──────────────────────────────────────────────

class TestProtectedArtifacts:
    def test_hash_manifest_exists(self):
        path = ROOT / "ml/experiments/phase10e_certification/hashes/phase10e_protected_artifacts_post_sha256.json"
        assert path.exists()

    def test_protected_artifacts_no_drift(self, validator):
        result = validator.check_protected_artifacts()
        assert result["drift_count"] == 0, f"Protected artifact drift detected: {result['failed_artifacts']}"
        assert result["total_audited"] == CERTIFIED_PROTECTED_COUNT

    def test_protected_artifacts_pass(self, validator):
        result = validator.check_protected_artifacts()
        assert result["status"] == "PASS"


# ── Clean environment load ────────────────────────────────────────────────────

class TestCleanLoad:
    def test_service_loads(self, service):
        assert service.model is not None
        assert service.scaler is not None

    def test_calibration_loads(self, service):
        assert service.calibration_bias is not None
        assert isinstance(service.calibration_bias, float)

    def test_uncertainty_loads(self, service):
        assert service.bound_90 > 0

    def test_model_id_matches(self, service):
        assert service.model_config.get("model_id") == CERTIFIED_RELEASE_ID

    def test_sequence_contract(self, service):
        assert service.model_config.get("sequence_window") == CERTIFIED_WINDOW

    def test_feature_contract(self, service):
        assert service.model_config.get("feature_dimension") == CERTIFIED_FEATURE_DIM


# ── API smoke ─────────────────────────────────────────────────────────────────

class TestAPISMoke:
    def test_health_endpoint(self, service):
        result = service.health_endpoint()
        assert result.get("status") == "HEALTHY"
        assert result.get("model_loaded") is True

    def test_readiness_endpoint(self, service):
        result = service.readiness_endpoint()
        assert result.get("status") == "READY"
        assert result.get("scaler_ready") is True
        assert result.get("calibration_ready") is True

    def test_version_endpoint(self, service):
        result = service.version_endpoint()
        assert result.get("model_id") == CERTIFIED_RELEASE_ID

    def test_predict_endpoint_returns_forecasts(self, service, known_good_fixture):
        payload = {"records": known_good_fixture.to_dict(orient="records")}
        result = service.predict_endpoint(payload)
        assert result.get("status") == "SUCCESS"
        forecasts = result.get("forecasts", [])
        assert len(forecasts) > 0
        assert "forecast_pm25" in forecasts[0]

    def test_predict_returns_conformal_bounds(self, service, known_good_fixture):
        payload = {"records": known_good_fixture.to_dict(orient="records")}
        result = service.predict_endpoint(payload)
        f = result["forecasts"][0]
        assert "lower_90" in f
        assert "upper_90" in f
        assert "conformal_half_width" in f

    def test_predict_provenance(self, service, known_good_fixture):
        payload = {"records": known_good_fixture.to_dict(orient="records")}
        result = service.predict_endpoint(payload)
        assert result.get("model_version") == CERTIFIED_RELEASE_ID


# ── Deterministic inference ───────────────────────────────────────────────────

class TestDeterministicInference:
    def test_repeated_predictions_identical(self, service, known_good_fixture):
        payload = {"records": known_good_fixture.to_dict(orient="records")}
        preds = []
        for _ in range(5):
            r = service.predict_endpoint(payload)
            preds.append(r["forecasts"][0]["forecast_pm25"])

        deltas = [abs(p - preds[0]) for p in preds[1:]]
        max_delta = max(deltas)
        assert max_delta <= DETERMINISM_TOLERANCE, (
            f"Non-deterministic inference: max_delta={max_delta} > {DETERMINISM_TOLERANCE}"
        )


# ── Contract rejection ────────────────────────────────────────────────────────

class TestContractRejection:
    def test_missing_records_key_rejected(self, service):
        with pytest.raises(Exception):
            service.predict_endpoint({"wrong_key": []})

    def test_wrong_feature_dimension_rejected(self, service, known_good_fixture):
        truncated = known_good_fixture.drop(columns=[known_good_fixture.columns[0]])
        with pytest.raises(Exception):
            service.predict_endpoint({"records": truncated.to_dict(orient="records")})

    def test_nan_input_rejected(self, service, known_good_fixture):
        nan_df = known_good_fixture.copy()
        nan_df.iloc[0, 0] = float("nan")
        with pytest.raises(Exception):
            service.predict_endpoint({"records": nan_df.to_dict(orient="records")})


# ── Monitoring config ─────────────────────────────────────────────────────────

class TestMonitoringConfig:
    def test_model_registry_loads(self):
        path = ROOT / "ml/experiments/phase10b_observability/manifests/phase10b_model_registry.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data  # non-empty

    def test_rollback_policy_loads_or_is_conditional(self):
        path = ROOT / "ml/experiments/phase10b_observability/manifests/phase10b_rollback_policy.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            assert data

    def test_certified_candidate_in_registry(self):
        path = ROOT / "ml/experiments/phase10b_observability/manifests/phase10b_model_registry.json"
        if path.exists():
            content = path.read_text()
            assert CERTIFIED_RELEASE_ID in content or "TCN_CAL07_25" in content


# ── Performance smoke ─────────────────────────────────────────────────────────

class TestPerformanceSmoke:
    def test_single_inference_under_sla(self, service, known_good_fixture):
        import time
        payload = {"records": known_good_fixture.to_dict(orient="records")}
        # warm-up
        service.predict_endpoint(payload)
        t0 = time.perf_counter()
        service.predict_endpoint(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 10.0, f"Single inference {elapsed_ms:.2f} ms exceeds 10 ms SLA"

    def test_batch_under_sla(self, service, known_good_fixture):
        import time
        payload = {"records": known_good_fixture.to_dict(orient="records")}
        # Warm-up
        service.predict_endpoint(payload)
        # Single full pipeline call SLA = 50 ms
        t0 = time.perf_counter()
        service.predict_endpoint(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, f"Pipeline call {elapsed_ms:.2f} ms exceeds 50 ms SLA"


# ── Provenance ────────────────────────────────────────────────────────────────

class TestProvenance:
    def test_prediction_carries_model_version(self, service, known_good_fixture):
        payload = {"records": known_good_fixture.to_dict(orient="records")}
        result = service.predict_endpoint(payload)
        assert result.get("model_version") == CERTIFIED_RELEASE_ID

    def test_version_endpoint_distinct_from_research_model(self, service):
        v = service.version_endpoint()
        model_id = v.get("model_id", "")
        assert "50" not in model_id, "Research stress-test model (50%) must not appear in production"
        assert CERTIFIED_RELEASE_ID in model_id
