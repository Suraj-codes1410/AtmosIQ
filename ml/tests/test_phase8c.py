"""
Unit and Integration Tests for AtmosIQ Phase 8C (Corpus Consolidation & Release).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8c import (
    ReleaseConfigPhase8C,
    ExtremeTailGovernanceEngine,
    CorpusConsolidationEngine,
    Phase8CProvenanceManager,
    IntegrityAndIsolationAuditor,
    SyntheticAugmentationPolicyEngine,
    Phase9TrainingContractEngine,
)


class TestPhase8C:
    @classmethod
    def setup_class(cls):
        cls.config = ReleaseConfigPhase8C()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()

        cls.prov_mgr = Phase8CProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.auditor = IntegrityAndIsolationAuditor(cls.feature_registry, cls.config.locked_eval_start_date)
        cls.consolidation_engine = CorpusConsolidationEngine(cls.config, cls.feature_registry)

    # 1. Protected Artifact Immutability (Phase 6F Freeze Gate)
    def test_phase6f_freeze_gate(self):
        passed, summary = self.prov_mgr.verify_phase6f_freeze()
        assert passed is True
        assert summary["freeze_status"] == "PASS"
        assert summary["total_protected_artifacts"] == 21

    # 2. Extreme-Tail Governance Filter
    def test_extreme_tail_governance(self):
        gov_engine = ExtremeTailGovernanceEngine(250.0, 4500.0, 2.0)
        df_dummy = pd.DataFrame({
            "trajectory_id": ["T1", "T1", "T2", "T2"],
            "synthetic_date": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
            "pm25": [100.0, 280.0, 50.0, 60.0],
            "ventilation_index_1d": [2000.0, 5000.0, 1500.0, 1600.0], # T1 violates VI threshold during extreme PM
            "rainfall_1d": [0.0, 0.0, 0.0, 0.0],
        })
        df_comp, df_aud, sum_dict = gov_engine.audit_and_filter_corpus(df_dummy)
        assert sum_dict["rejected_trajectories"] == 1
        assert sum_dict["accepted_trajectories"] == 1
        assert "T1" not in df_comp["trajectory_id"].values
        assert "T2" in df_comp["trajectory_id"].values

    # 3. Provenance Manifest Generation
    def test_provenance_manifest_generation(self):
        df_dummy = pd.DataFrame({
            "trajectory_id": ["T1", "T1"],
            "batch_id": ["batch_0001", "batch_0001"],
            "generator_version": ["HP-STG-v1.0.0", "HP-STG-v1.0.0"],
            "generation_seed": [42, 42],
            "horizon_days": [14, 14],
            "synthetic_date": ["2025-01-01", "2025-01-02"],
        })
        df_prov = self.prov_mgr.generate_provenance_manifest(df_dummy, "v1.0.0")
        assert len(df_prov) == 2
        assert "provenance_hash" in df_prov.columns
        assert "observation_id" in df_prov.columns
        assert (df_prov["synthetic"] == True).all()

    # 4. Data Isolation Audit
    def test_data_isolation_audit(self):
        df_clean = pd.DataFrame({
            "synthetic_date": ["2021-05-01", "2021-05-02"],
            "data_origin": ["synthetic", "synthetic"],
            "source_partition": ["2020-2021", "2020-2021"],
        })
        passed, df_aud, sum_dict = self.auditor.audit_data_isolation(df_clean)
        assert passed is True
        assert sum_dict["total_isolation_violations"] == 0

        # Leakage test
        df_leaked = pd.DataFrame({
            "date": ["2022-05-01"],
            "data_origin": ["synthetic"],
            "source_partition": ["2020-2021"],
        })
        passed_leak, _, sum_leak = self.auditor.audit_data_isolation(df_leaked)
        assert passed_leak is False
        assert sum_leak["total_isolation_violations"] == 1

    # 5. Dataset Integrity & Hydrodynamic Identity
    def test_corpus_integrity_audit(self):
        # Create a valid minimal dataframe matching feature registry
        data = {f: [10.0, 20.0] for f in self.feature_registry}
        data["trajectory_id"] = ["T1"] * 14
        data["synthetic_date"] = [f"2025-01-{i+1:02d}" for i in range(14)]
        # expand rows to 14
        df_valid = pd.DataFrame({f: [10.0] * 14 for f in self.feature_registry})
        df_valid["trajectory_id"] = "T1"
        df_valid["synthetic_date"] = [f"2025-01-{i+1:02d}" for i in range(14)]
        df_valid["pm25"] = 50.0
        df_valid["wind_speed_kmh"] = 18.0
        df_valid["pblh_1d"] = 1000.0
        # VI = (18 * 1000 / 3600) * 1000 = 5 * 1000 = 5000.0
        df_valid["ventilation_index_1d"] = 5000.0
        df_valid["rainfall_1d"] = 0.0
        df_valid["rain_event_1d"] = 0

        passed, df_aud, sum_dict = self.auditor.audit_corpus_integrity(df_valid)
        assert passed is True
        assert sum_dict["failed_checks"] == 0

    # 6. Synthetic Augmentation Policy
    def test_synthetic_augmentation_policy(self, tmp_path):
        policy_eng = SyntheticAugmentationPolicyEngine()
        policy_file = tmp_path / "synthetic_augmentation_policy.json"
        p_data = policy_eng.generate_policy_file(policy_file)
        assert policy_file.exists()
        assert p_data["recommended_ratio"] == 0.25
        assert p_data["maximum_ratio"] == 0.50
        assert 1.00 in p_data["prohibited_ratios"]

    # 7. Phase 9 Training Contract
    def test_phase9_training_contract(self, tmp_path):
        contract_eng = Phase9TrainingContractEngine(self.feature_registry)
        contract_file = tmp_path / "phase9_training_contract.json"
        c_data = contract_eng.generate_contract(
            corpus_path=tmp_path / "corpus.parquet",
            corpus_sha256="test_sha",
            output_path=contract_file
        )
        assert contract_file.exists()
        assert c_data["issuing_phase"] == "Phase 8C"
        assert c_data["augmentation_rules"]["default_recommended_ratio"] == 0.25
        assert c_data["feature_contract"]["required_feature_count"] == len(self.feature_registry)

    # 8. Reproducibility Auditor
    def test_reproducibility_audit(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        is_repro, df_aud, sum_dict = self.auditor.audit_reproducibility(df1, df1)
        assert is_repro is True
        assert sum_dict["maximum_numerical_delta"] <= 1e-9
