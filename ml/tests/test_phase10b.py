"""
Unit and Integration Tests for AtmosIQ Phase 10B (Production Observability, Drift Monitoring, Alerting & Governance).
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase10b import (
    Phase10BConfig,
    Phase10BProvenanceManager,
    Phase10BDriftMonitor,
    Phase10BAlertingEngine,
    Phase10BMonitoringStressTester,
    Phase10BRegistryManager,
)


class TestPhase10B:
    @classmethod
    def setup_class(cls):
        cls.config = Phase10BConfig()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()
        cls.prov_mgr = Phase10BProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.drift_mon = Phase10BDriftMonitor(cls.feature_registry)
        cls.alert_engine = Phase10BAlertingEngine(cls.config.manifests_dir)
        cls.stress_tester = Phase10BMonitoringStressTester(cls.alert_engine, cls.drift_mon)
        cls.registry_mgr = Phase10BRegistryManager(cls.config.manifests_dir, cls.config.benchmarks_dir)

    # 1. Protected Artifacts Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Population Stability Index (PSI) Calculation
    def test_psi_calculation(self):
        np.random.seed(42)
        base = np.random.normal(50.0, 10.0, size=500)
        target_identical = np.random.normal(50.0, 10.0, size=500)
        target_shifted = np.random.normal(70.0, 15.0, size=500)

        psi_ident = self.drift_mon.calculate_psi(base, target_identical)
        psi_shift = self.drift_mon.calculate_psi(base, target_shifted)

        assert psi_ident < 0.10
        assert psi_shift > 0.25

    # 3. Physical Sanity Checks
    def test_physical_sanity_checks(self):
        df_valid = pd.DataFrame({
            "pm25": [10.0, 50.0, 100.0],
            "relative_humidity_2m": [45.0, 60.0, 80.0],
            "wind_speed_10m": [2.0, 3.0, 4.0],
            "boundary_layer_height": [500.0, 600.0, 700.0],
            "ventilation_index": [1000.0, 1800.0, 2800.0],
        })
        df_res = self.drift_mon.monitor_physical_sanity(df_valid)
        assert (df_res["status"] == "PASS").all()

        df_invalid = pd.DataFrame({
            "pm25": [-5.0, 50.0], # Negative PM2.5 violation
            "relative_humidity_2m": [120.0, 50.0], # RH > 100% violation
        })
        df_res_inv = self.drift_mon.monitor_physical_sanity(df_invalid)
        assert (df_res_inv["violation_count"] > 0).any()

    # 4. Alert Severity Framework
    def test_alert_severity_evaluation(self):
        # Normal Telemetry -> 0 Alerts (GREEN)
        alerts_normal = self.alert_engine.evaluate_telemetry(
            mae_current=34.0, mae_baseline=33.62, bias_current=-2.5,
            psi_max=0.08, coverage_90=0.91, contract_violations_count=0,
            model_version="TEST_v1.0.0"
        )
        assert len(alerts_normal) == 0

        # Critical Telemetry -> Contract Violation (RED)
        alerts_critical = self.alert_engine.evaluate_telemetry(
            mae_current=34.0, mae_baseline=33.62, bias_current=-2.5,
            psi_max=0.08, coverage_90=0.91, contract_violations_count=1,
            model_version="TEST_v1.0.0"
        )
        assert len(alerts_critical) > 0
        assert any(a["severity"] == "RED" for a in alerts_critical)

    # 5. Monitoring Stress Chaos Testing
    def test_monitoring_chaos_stress_tests(self):
        df_stress = self.stress_tester.run_all_stress_scenarios(baseline_mae=33.62)
        assert len(df_stress) == 10
        assert (df_stress["detection_status"] == "PASS_DETECTED").all()

    # 6. Model Registry & Policy Export
    def test_model_registry_and_policy_export(self):
        reg_path = self.registry_mgr.export_model_registry("test_hash_123")
        contract_path = self.registry_mgr.export_monitoring_contract()
        alert_path, rollback_path = self.alert_engine.export_alert_and_rollback_policies("TEST_v1.0.0")

        assert reg_path.exists()
        assert contract_path.exists()
        assert alert_path.exists()
        assert rollback_path.exists()
