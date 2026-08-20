"""
Unit and Integration Tests for AtmosIQ Phase 10D (Final Production Release & Deployment Certification).
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase10d import (
    Phase10DConfig,
    Phase10DProvenanceManager,
    Phase10DReleaseManager,
    Phase10DDeploymentService,
    ServiceContractException,
    Phase10DGovernanceValidator,
)
from ml.src.modeling.phase8g.sequence_builder import Phase8GSequenceBuilder


class TestPhase10D:
    @classmethod
    def setup_class(cls):
        cls.config = Phase10DConfig()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()
        cls.prov_mgr = Phase10DProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.seq_builder = Phase8GSequenceBuilder(cls.feature_registry, "pm25")

        df_full = pd.read_csv(cls.config.dataset_v3_path)
        df_dev = df_full[(df_full["date"] >= "2020-01-01") & (df_full["date"] <= "2021-12-31")]
        cls.seq_builder.fit_scaler(df_dev)

        cls.rel_mgr = Phase10DReleaseManager(cls.config)
        prod_ckpt_path = cls.config.phase9_checkpoints_dir / "checkpoint_TCN_aug25pct_seed2025.json"
        cls.rel_mgr.build_release_bundle(
            checkpoint_path=prod_ckpt_path,
            scaler=cls.seq_builder.scaler,
            feature_registry=cls.feature_registry,
        )

        cls.service = Phase10DDeploymentService(cls.config.bundle_dir)
        cls.gov_validator = Phase10DGovernanceValidator(cls.config, cls.config.bundle_dir)
        cls.df_sample = df_dev.iloc[:28].copy()

    # 1. Protected Artifacts Freeze (32 Artifacts)
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Deployed Service Endpoints
    def test_service_endpoints(self):
        h = self.service.health_endpoint()
        assert h["status"] == "HEALTHY"
        assert h["model_loaded"] is True

        r = self.service.readiness_endpoint()
        assert r["status"] == "READY"
        assert r["scaler_ready"] is True
        assert r["calibration_ready"] is True

        v = self.service.version_endpoint()
        assert v["model_id"] == self.config.production_release_id
        assert v["release_status"] == "RELEASE_CERTIFIED"

    # 3. Predict Endpoint & Deployed Equivalence
    def test_predict_endpoint_and_equivalence(self):
        payload = {"records": self.df_sample.to_dict(orient="records")}
        res = self.service.predict_endpoint(payload)
        assert res["status"] == "SUCCESS"
        assert len(res["forecasts"]) == len(self.df_sample) - 14 + 1
        assert res["execution_latency_ms"] >= 0.0

        f0 = res["forecasts"][0]
        assert "prediction_id" in f0
        assert f0["lower_90"] <= f0["forecast_pm25"] <= f0["upper_90"]

    # 4. Service Rejection of Invalid Requests
    def test_service_invalid_requests(self):
        with pytest.raises(ServiceContractException):
            self.service.predict_endpoint({"invalid": []})

        with pytest.raises(ServiceContractException):
            self.service.predict_endpoint({"records": self.df_sample.iloc[:5].to_dict(orient="records")})

        with pytest.raises(ServiceContractException):
            self.service.predict_endpoint({"records": self.df_sample.drop(columns=[self.feature_registry[0]]).to_dict(orient="records")})

    # 5. Rollback & Restart Recovery
    def test_rollback_and_restart_recovery(self):
        df_rollback = self.gov_validator.run_rollback_drill()
        assert len(df_rollback) == 6
        assert (df_rollback["status"].str.contains("ACTIVE|TRIGGERED|INITIATED|VERIFIED|READY|PASS")).all()

        df_restart = self.gov_validator.run_restart_recovery_test(self.df_sample)
        assert df_restart["status"].iloc[0] == "PASS_ZERO_DRIFT"
        assert df_restart["max_numerical_delta"].iloc[0] <= 1e-9

    # 6. Deployment Chaos Suite (16 Scenarios)
    def test_deployment_chaos_suite(self):
        df_chaos = self.gov_validator.run_deployment_chaos_suite(self.df_sample)
        assert len(df_chaos) == 16
        assert (df_chaos["status"] == "PASS").all()

    # 7. Security & Secret Scanning Audit
    def test_security_audit(self):
        df_sec = self.gov_validator.run_security_and_config_audit()
        assert len(df_sec) == 5
        assert (df_sec["status"] == "PASS").all()
