"""
Unit and Integration Tests for AtmosIQ Phase 8A (Generation Infrastructure).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8a import (
    GenerationConfigPhase8A,
    EvaluationFirewall,
    EvaluationFirewallViolation,
    Phase8AProvenanceManager,
    Phase8APhysicsValidator,
    ExtremeTailFilter,
    OODSupportScorer,
    MemorizationScreen,
    ProductionTrajectoryGenerator,
    DatasetSharder,
    DatasetManifestGenerator,
)


class TestPhase8A:
    @classmethod
    def setup_class(cls):
        cls.config = GenerationConfigPhase8A(mode="PILOT")
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()

        df_full = pd.read_csv(cls.config.dataset_v3_path)
        cls.df_real_train = df_full[
            (df_full["date"] >= cls.config.dev_train_start_date) &
            (df_full["date"] <= cls.config.dev_train_end_date)
        ].copy()

        def classify_season(m):
            if m in [12, 1, 2]: return "Winter"
            if m in [3, 4, 5]: return "Summer"
            if m in [6, 7, 8, 9]: return "Monsoon"
            return "Post-Monsoon"

        def classify_regime(pm):
            if pm < 60.0: return "Low"
            if pm < 120.0: return "Moderate"
            if pm < 250.0: return "High"
            return "Extreme"

        cls.df_real_train["month"] = pd.to_datetime(cls.df_real_train["date"]).dt.month
        cls.df_real_train["season"] = cls.df_real_train["month"].apply(classify_season)
        cls.df_real_train["pollution_regime"] = cls.df_real_train["pm25"].apply(classify_regime)

        cls.generator = ProductionTrajectoryGenerator(cls.config, cls.feature_registry)
        cls.generator.fit_from_training_data(cls.df_real_train)

    # 1. Configuration Validation & Horizon Constraints
    def test_configuration_valid_horizons(self):
        cfg = GenerationConfigPhase8A(trajectory_lengths=[14, 30])
        assert cfg.trajectory_lengths == [14, 30]

        with pytest.raises(ValueError, match="Unsupported trajectory horizon"):
            GenerationConfigPhase8A(trajectory_lengths=[7, 14])

        with pytest.raises(ValueError, match="Unsupported trajectory horizon"):
            GenerationConfigPhase8A(trajectory_lengths=[60])

    # 2. Configuration Augmentation Ratio Constraints
    def test_configuration_augmentation_ratios(self):
        cfg = GenerationConfigPhase8A(augmentation_ratio=0.25)
        assert cfg.augmentation_ratio == 0.25

        with pytest.raises(ValueError, match="Unsupported augmentation ratio"):
            GenerationConfigPhase8A(augmentation_ratio=1.00)

        with pytest.raises(ValueError, match="Unsupported augmentation ratio"):
            GenerationConfigPhase8A(augmentation_ratio=0.05)

    # 3. Data Isolation Firewall
    def test_evaluation_firewall(self):
        firewall = EvaluationFirewall("2022-01-01")
        assert firewall.verify_training_partition_isolation(self.df_real_train) is True

        # Test leakage detection
        df_leaked = pd.DataFrame({"date": ["2021-12-31", "2022-01-01", "2023-05-15"], "pm25": [50, 60, 70]})
        with pytest.raises(EvaluationFirewallViolation, match="CRITICAL FIREWALL VIOLATION"):
            firewall.verify_training_partition_isolation(df_leaked, "leaked_dataset")

    # 4. Phase 6F Freeze Gate Verification
    def test_freeze_gate(self):
        prov_mgr = Phase8AProvenanceManager(self.root_dir, self.config.freeze_manifest_path)
        passed, summary = prov_mgr.verify_phase6f_freeze()
        assert passed is True
        assert summary["freeze_status"] == "PASS"
        assert summary["total_protected_artifacts"] == 21

    # 5. Deterministic Trajectory Generation & Seed Derivation
    def test_deterministic_seed_derivation(self):
        s1 = Phase8AProvenanceManager.derive_trajectory_seed(42, "SYNTH_001")
        s2 = Phase8AProvenanceManager.derive_trajectory_seed(42, "SYNTH_001")
        s3 = Phase8AProvenanceManager.derive_trajectory_seed(42, "SYNTH_002")
        assert s1 == s2
        assert s1 != s3

        # Trajectory generation determinism
        df1 = self.generator.generate_single_trajectory("TEST_T1", 14, "Winter", s1)
        df2 = self.generator.generate_single_trajectory("TEST_T1", 14, "Winter", s1)
        assert np.allclose(df1["pm25"].values, df2["pm25"].values)

    # 6. Schema Compatibility (35 prediction-safe features)
    def test_schema_compatibility(self):
        s = Phase8AProvenanceManager.derive_trajectory_seed(42, "TEST_SCHEMA")
        df_traj = self.generator.generate_single_trajectory("TEST_SCHEMA", 14, "Winter", s)
        for feat in self.feature_registry:
            assert feat in df_traj.columns
        assert "trajectory_id" in df_traj.columns
        assert "data_origin" in df_traj.columns
        assert (df_traj["data_origin"] == "synthetic").all()

    # 7. Physics Validation Engine
    def test_physics_validation(self):
        val = Phase8APhysicsValidator()
        s = Phase8AProvenanceManager.derive_trajectory_seed(42, "TEST_PHYS")
        df_traj = self.generator.generate_single_trajectory("TEST_PHYS", 14, "Winter", s)
        is_ok, report = val.validate_trajectory(df_traj, "TEST_PHYS")
        assert is_ok is True
        assert report["violation_count"] == 0

    # 8. Ventilation Index Hydrodynamic Identity
    def test_hydrodynamic_ventilation_identity(self):
        s = Phase8AProvenanceManager.derive_trajectory_seed(42, "TEST_VI")
        df_traj = self.generator.generate_single_trajectory("TEST_VI", 14, "Winter", s)
        for _, row in df_traj.iterrows():
            ws_ms = row["wind_speed_kmh"] * (1000.0 / 3600.0)
            expected_vi = ws_ms * row["pblh_1d"]
            assert np.isclose(row["ventilation_index_1d"], expected_vi, rtol=1e-3)

    # 9. Extreme-Tail Environmental Filtering (Restriction C)
    def test_extreme_tail_filtering(self):
        filter_engine = ExtremeTailFilter(
            enabled=True,
            extreme_pm25_threshold=250.0,
            vi_threshold=4500.0,
            precipitation_threshold=2.0
        )
        # Create an artificial incoherent trajectory
        df_bad = pd.DataFrame({
            "synthetic_date": ["2025-01-01"],
            "pm25": [300.0],
            "ventilation_index_1d": [5500.0], # Incoherent high VI
            "rainfall_1d": [0.0],
            "season": ["Winter"]
        })
        is_acc, rejections = filter_engine.evaluate_trajectory(df_bad, "BAD_TRAJ", 42, "HP-STG-v1.0.0")
        assert is_acc is False
        assert len(rejections) == 1
        assert "VI" in rejections[0].rejection_reason

    # 10. OOD Support Annotation
    def test_ood_support_annotation(self):
        s = Phase8AProvenanceManager.derive_trajectory_seed(42, "TEST_OOD")
        df_traj = self.generator.generate_single_trajectory("TEST_OOD", 14, "Winter", s)
        assert "ood_distance" in df_traj.columns
        assert "max_feature_zscore" in df_traj.columns
        assert "is_ood_flag" in df_traj.columns
        assert not df_traj["ood_distance"].isna().any()

    # 11. Memorization Screening
    def test_memorization_screening(self):
        screen = MemorizationScreen(self.feature_registry)
        screen.fit(self.df_real_train)
        s = Phase8AProvenanceManager.derive_trajectory_seed(42, "TEST_MEM")
        df_traj = self.generator.generate_single_trajectory("TEST_MEM", 14, "Winter", s)
        passed, report = screen.screen_trajectory(df_traj, "TEST_MEM")
        assert passed is True
        assert report["exact_duplicates"] == 0
        assert report["near_duplicates"] == 0

    # 12. Sharding Engine
    def test_dataset_sharder(self, tmp_path):
        sharder = DatasetSharder(tmp_path / "shards", max_trajectories_per_shard=2, config_hash="test_hash")
        trajs = [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
            pd.DataFrame({"a": [9, 10], "b": [11, 12]}),
        ]
        shard_recs, df_cons = sharder.write_shards(trajs)
        assert len(shard_recs) == 2
        assert len(df_cons) == 6
        assert (tmp_path / "shards" / "shard-000001.parquet").exists()
        assert (tmp_path / "shards" / "shard-000002.parquet").exists()

    # 13. Manifest Generation
    def test_manifest_generator(self, tmp_path):
        manifest_gen = DatasetManifestGenerator(tmp_path / "manifests")
        cfg_dict = self.config.to_dict()
        m_data = manifest_gen.generate_manifest(
            config_dict=cfg_dict,
            source_dataset_sha256="test_src_sha",
            shard_records=[{"shard_name": "shard-000001.parquet", "sha256": "1234", "observation_count": 10}],
            generation_stats={"accepted_trajectories": 1, "rejected_trajectories": 0, "accepted_observations": 10, "acceptance_rate_pct": 100.0},
            df_rejections=pd.DataFrame()
        )
        assert (tmp_path / "manifests" / "dataset_manifest.json").exists()
        assert (tmp_path / "manifests" / "provenance.json").exists()
        assert m_data["dataset_version"] == "AtmosIQ-SYNTH-v8A"

    # 14. Pilot Generation Batch Execution
    def test_pilot_generation_batch(self):
        specs = [(14, "Winter"), (30, "Summer")]
        accepted, rejected, stats = self.generator.generate_batch(specs)
        assert stats["requested_trajectories"] == 2
        assert stats["accepted_trajectories"] >= 1
        assert stats["accepted_observations"] >= 14
