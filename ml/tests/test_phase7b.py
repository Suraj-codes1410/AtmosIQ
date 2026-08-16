"""
Unit and Integration Tests for AtmosIQ Phase 7B (HP-STG & Physics Constraints).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase7b import (
    SyntheticConfigPhase7B,
    ProvenanceVerifierPhase7B,
    TrajectoryGeneratorPhase7B,
    PhysicsConstraintEnginePhase7B,
    ExtremeEventGenerator,
    FeatureReconstructorPhase7B,
    ValidationPrecheckerPhase7B,
    ReproducibilityAuditorPhase7B,
)


class TestPhase7B:
    @classmethod
    def setup_class(cls):
        cls.config = SyntheticConfigPhase7B()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()

        # Load authorized training dataset
        df_full = pd.read_csv(cls.config.dataset_v3_path)
        cls.df_train = df_full[
            (df_full["date"] >= cls.config.training_start_date) &
            (df_full["date"] <= cls.config.training_end_date)
        ].copy()

        cls.generator = TrajectoryGeneratorPhase7B(cls.config, cls.feature_registry)
        cls.generator.fit_from_training_data(cls.df_train)
        cls.df_sample_traj = cls.generator.generate_single_trajectory("TEST_T001", 14, "Winter")

    # 1. Provenance & Freeze Gate
    def test_provenance_and_freeze(self):
        verifier = ProvenanceVerifierPhase7B(self.root_dir)
        spec_pass, spec_hash = verifier.verify_phase7a_spec()
        assert spec_pass is True
        assert len(spec_hash) == 64

        freeze_pass, violations = verifier.verify_phase6f_freeze()
        assert freeze_pass is True
        assert len(violations) == 0

    # 2. Hard Non-Negativity Constraints
    def test_physical_non_negativity(self):
        assert (self.df_sample_traj["pm25"] >= 0.0).all()
        assert (self.df_sample_traj["wind_speed_kmh"] >= 0.0).all()
        assert (self.df_sample_traj["rainfall_1d"] >= 0.0).all()
        assert (self.df_sample_traj["pblh_1d"] >= 150.0).all()
        assert (self.df_sample_traj["fire_hotspot_count_1d"] >= 0.0).all()

    # 3. Boundary Layer & Climatological Bounds
    def test_physical_bounds(self):
        assert (self.df_sample_traj["humidity_pct"] >= 5.0).all()
        assert (self.df_sample_traj["humidity_pct"] <= 100.0).all()
        assert (self.df_sample_traj["temperature_c"] >= 0.0).all()
        assert (self.df_sample_traj["temperature_c"] <= 50.0).all()

    # 4. Hydrodynamic Ventilation Index Consistency
    def test_ventilation_index_consistency(self):
        for _, row in self.df_sample_traj.iterrows():
            ws_ms = row["wind_speed_kmh"] * (1000.0 / 3600.0)
            expected_vi = ws_ms * row["pblh_1d"]
            assert np.isclose(row["ventilation_index_1d"], expected_vi, rtol=1e-3)

    # 5. Precipitation Binary Indicator
    def test_rain_event_binary_logic(self):
        for _, row in self.df_sample_traj.iterrows():
            if row["rainfall_1d"] >= 1.0:
                assert row["rain_event_1d"] == 1
            else:
                assert row["rain_event_1d"] == 0

    # 6. Mathematical Lag Consistency
    def test_lag_consistency(self):
        pm_series = self.df_sample_traj["pm25"].values
        for t in range(1, len(pm_series)):
            assert np.isclose(self.df_sample_traj["pm25_lag_1d"].iloc[t], pm_series[t-1])

    # 7. Mathematical Rolling Feature Consistency
    def test_rolling_consistency(self):
        pm_series = self.df_sample_traj["pm25"].values
        for t in range(2, len(pm_series)):
            expected_roll3 = np.mean(pm_series[t-2:t+1])
            assert np.isclose(self.df_sample_traj["pm25_roll_mean_3d"].iloc[t], expected_roll3)

        # Min <= Mean <= Max identity
        assert (self.df_sample_traj["pm25_roll_min_7d"] <= self.df_sample_traj["pm25_roll_mean_7d"] + 1e-5).all()
        assert (self.df_sample_traj["pm25_roll_mean_7d"] <= self.df_sample_traj["pm25_roll_max_7d"] + 1e-5).all()

    # 8. Schema Completeness (All 35 prediction-safe features present)
    def test_schema_completeness(self):
        for feat in self.feature_registry:
            assert feat in self.df_sample_traj.columns

        assert "pm25" in self.df_sample_traj.columns
        assert "synthetic_date" in self.df_sample_traj.columns
        assert "trajectory_id" in self.df_sample_traj.columns

    # 9. Provenance Metadata
    def test_provenance_fields(self):
        assert (self.df_sample_traj["data_origin"] == "synthetic").all()
        assert (self.df_sample_traj["generator_version"] == "HP-STG v1.0.0").all()
        assert (self.df_sample_traj["random_seed"] == 42).all()
        assert self.df_sample_traj["generation_timestamp"].notna().all()

    # 10. NaN and Inf Prevention
    def test_nan_inf_prevention(self):
        eval_cols = self.feature_registry + ["pm25"]
        assert self.df_sample_traj[eval_cols].isna().sum().sum() == 0
        assert not np.isinf(self.df_sample_traj[eval_cols].values).any()

    # 11. Deterministic Reproducibility
    def test_reproducibility(self):
        auditor = ReproducibilityAuditorPhase7B(self.config, self.feature_registry)
        is_repro, max_delta, df_delta = auditor.run_reproducibility_audit(self.df_train)
        assert is_repro is True
        assert max_delta <= 1e-10

    # 12. Extreme Event Coherence
    def test_extreme_event_coherence(self):
        extreme_gen = ExtremeEventGenerator()
        res = extreme_gen.evaluate_extreme_coherence(self.df_sample_traj)
        assert res["coherence_rate"] >= 0.90
