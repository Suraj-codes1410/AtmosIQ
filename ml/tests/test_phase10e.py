"""
Unit and Integration Tests for AtmosIQ Phase 10E (Final Production Certification).
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase10e import (
    Phase10EConfig,
    Phase10EEvidenceIndexer,
    Phase10EIntegrityAuditor,
    Phase10ELineageAuditor,
    Phase10EDomainAuditor,
    Phase10ECertificationGate,
)


class TestPhase10E:
    @classmethod
    def setup_class(cls):
        cls.config = Phase10EConfig()
        cls.root_dir = cls.config.root_dir
        cls.indexer = Phase10EEvidenceIndexer(cls.root_dir)
        cls.integrity_auditor = Phase10EIntegrityAuditor(cls.root_dir, cls.config.freeze_manifest_path)
        cls.lineage_auditor = Phase10ELineageAuditor(cls.config)
        cls.domain_auditor = Phase10EDomainAuditor(cls.config)
        cls.cert_gate = Phase10ECertificationGate(cls.config)

    # 1. Evidence Indexing Audit
    def test_evidence_indexing(self):
        index = self.indexer.build_evidence_index()
        assert index["total_artifacts_indexed"] >= 13
        assert index["all_critical_present"] is True

    # 2. Protected Artifacts Freeze (33 Artifacts)
    def test_protected_artifacts_freeze(self):
        passed, df_audit, summary = self.integrity_auditor.audit_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0
        assert (df_audit["status"] == "PASS").all()

    # 3. Model Lineage & Consistency
    def test_lineage_and_consistency(self):
        lineage_json, df_lineage = self.lineage_auditor.build_lineage_graph()
        assert lineage_json["production_release_id"] == self.config.production_release_id
        assert len(df_lineage) == 10

        df_cons = self.lineage_auditor.audit_cross_phase_consistency()
        assert len(df_cons) >= 10
        assert (df_cons["status"] == "PASS").all()

    # 4. Data Governance & Partition Firewall
    def test_data_governance(self):
        df_gov = self.domain_auditor.audit_data_governance()
        assert len(df_gov) == 6
        assert (df_gov["status"] == "PASS").all()

    # 5. Performance, Calibration & Uncertainty
    def test_performance_and_uncertainty(self):
        df_perf, limitations_md = self.domain_auditor.audit_performance()
        assert len(df_perf) == 6
        assert "KNOWN_WEAKNESS_MONITORED" in df_perf["status"].values
        assert len(limitations_md) > 100

        df_unc = self.domain_auditor.audit_uncertainty_and_calibration()
        assert len(df_unc) == 5
        assert (df_unc["status"] == "PASS").all()

    # 6. Deployment & Governance Audits
    def test_deployment_and_governance(self):
        df_dep, df_obs, df_sec = self.domain_auditor.audit_deployment_and_governance()
        assert len(df_dep) == 4
        assert (df_dep["status"] == "PASS").all()
        assert (df_obs["status"].str.contains("PASS")).all()
        assert (df_sec["status"] == "PASS").all()

    # 7. Mandatory 22 Certification Gates & Master Decision
    def test_certification_gates_and_decision(self):
        decision, df_gates, manifest = self.cert_gate.evaluate_all_gates()
        assert decision == "FINAL_PRODUCTION_CERTIFIED"
        assert len(df_gates) == 22
        assert (df_gates["status"] == "PASS").all()
        assert manifest["certification_decision"] == "FINAL_PRODUCTION_CERTIFIED"
