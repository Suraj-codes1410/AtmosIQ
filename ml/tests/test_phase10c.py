"""
Unit and Integration Tests for AtmosIQ Phase 10C (End-to-End Production Inference Validation).
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase10c import (
    Phase10CConfig,
    Phase10CProvenanceManager,
    Phase10CProductionPipeline,
    ProductionInferenceException,
    Phase10CFailureInjector,
    Phase10CInferenceAuditor,
)
from ml.src.modeling.phase9.models import Phase9TCNModel
from ml.src.modeling.phase8g.sequence_builder import Phase8GSequenceBuilder


class TestPhase10C:
    @classmethod
    def setup_class(cls):
        cls.config = Phase10CConfig()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()
        cls.prov_mgr = Phase10CProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.seq_builder = Phase8GSequenceBuilder(cls.feature_registry, "pm25")

        df_full = pd.read_csv(cls.config.dataset_v3_path)
        df_dev = df_full[(df_full["date"] >= "2020-01-01") & (df_full["date"] <= "2021-12-31")]
        cls.seq_builder.fit_scaler(df_dev)

        cls.model = Phase9TCNModel(window_size=14, feature_dim=35, seed=2025)
        cls.pipeline = Phase10CProductionPipeline(
            model=cls.model,
            scaler=cls.seq_builder.scaler,
            feature_registry=cls.feature_registry,
            window_size=14,
            feature_dim=35,
            model_version=cls.config.production_model_id,
            model_hash="test_hash_123",
            calibration_bias=-5.06,
            conformal_bound_80=63.92,
            conformal_bound_90=95.66,
            conformal_bound_95=117.50,
        )
        cls.injector = Phase10CFailureInjector(cls.pipeline)
        cls.auditor = Phase10CInferenceAuditor(cls.pipeline, cls.feature_registry)
        cls.df_sample = df_dev.iloc[:28].copy()

    # 1. Protected Artifacts Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Pipeline Forward Pass & Structured Response
    def test_production_pipeline_predict(self):
        response = self.pipeline.predict(self.df_sample, batch_id="TEST_BATCH_001")
        assert "forecasts" in response
        assert len(response["forecasts"]) == len(self.df_sample) - 14 + 1
        assert response["batch_size"] == len(response["forecasts"])
        assert response["execution_latency_ms"] >= 0.0

        f0 = response["forecasts"][0]
        assert "prediction_id" in f0
        assert "forecast_pm25" in f0
        assert "uncertainty_intervals" in f0
        assert f0["uncertainty_intervals"]["conformal_90"]["lower"] <= f0["forecast_pm25"]
        assert f0["uncertainty_intervals"]["conformal_90"]["upper"] >= f0["forecast_pm25"]

    # 3. Input Validation & Schema Enforcement
    def test_input_validation(self):
        with pytest.raises(ProductionInferenceException):
            self.pipeline.validate_raw_dataframe(self.df_sample.iloc[:5]) # Insufficient length

        with pytest.raises(ProductionInferenceException):
            self.pipeline.validate_raw_dataframe(self.df_sample.drop(columns=[self.feature_registry[0]])) # Missing feature

    # 4. Failure Injection Suite (16 Cases)
    def test_failure_injection_suite(self):
        df_fail = self.injector.run_all_failure_injections(self.df_sample)
        assert len(df_fail) == 16
        assert (df_fail["status"] == "PASS").all()

    # 5. Replay Equivalence & Reproducibility
    def test_replay_equivalence_and_reproducibility(self):
        # Obtain predictions
        resp = self.pipeline.predict(self.df_sample)
        preds = np.array([f["forecast_pm25"] for f in resp["forecasts"]])

        df_replay = self.auditor.audit_replay_equivalence(self.df_sample, preds)
        assert df_replay["equivalence_status"].iloc[0] == "PASS_NUMERICALLY_IDENTICAL"
        assert df_replay["max_absolute_delta"].iloc[0] <= 1e-9

        df_reprod = self.auditor.audit_reproducibility(self.df_sample)
        assert df_reprod["status"].iloc[0] == "PASS"

    # 6. Forensic Leakage Audit
    def test_end_to_end_leakage_audit(self):
        df_leakage = self.auditor.audit_end_to_end_leakage()
        assert len(df_leakage) == 5
        assert (df_leakage["status"] == "PASS").all()
        assert not df_leakage["leakage_detected"].any()
