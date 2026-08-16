"""
AtmosIQ Phase 8A: Production-Grade Seeded Trajectory Generation Engine.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import logging

from ml.src.modeling.phase7b.config import SyntheticConfigPhase7B
from ml.src.modeling.phase7b.trajectory_generator import TrajectoryGeneratorPhase7B

from .config import GenerationConfigPhase8A
from .provenance import Phase8AProvenanceManager
from .validation import Phase8APhysicsValidator
from .filtering import ExtremeTailFilter
from .ood_support import OODSupportScorer
from .memorization import MemorizationScreen
from .firewall import EvaluationFirewall

logger = logging.getLogger(__name__)


class ProductionTrajectoryGenerator:
    """Production synthetic trajectory generator implementing HP-STG v1.0.0 with full pipeline checks."""

    def __init__(self, config: GenerationConfigPhase8A, feature_registry: List[str]):
        self.config = config
        self.feature_registry = list(feature_registry)

        # Underlying HP-STG v1.0.0 Generator
        synth_cfg = SyntheticConfigPhase7B(
            random_seed=self.config.global_seed,
            training_start_date=self.config.dev_train_start_date,
            training_end_date=self.config.dev_train_end_date,
            locked_test_start_date=self.config.locked_eval_start_date,
        )
        self.hp_stg_generator = TrajectoryGeneratorPhase7B(synth_cfg, self.feature_registry)

        # Production Phase 8A validators & filters
        self.firewall = EvaluationFirewall(self.config.locked_eval_start_date)
        self.physics_validator = Phase8APhysicsValidator()
        self.extreme_filter = ExtremeTailFilter(
            enabled=self.config.extreme_filter_enabled,
            extreme_pm25_threshold=self.config.extreme_pm25_threshold,
            vi_threshold=self.config.vi_threshold,
            precipitation_threshold=self.config.precipitation_threshold,
        )
        self.ood_scorer = OODSupportScorer(self.feature_registry)
        self.memorization_screen = MemorizationScreen(self.feature_registry)

        self._is_fit = False

    def fit_from_training_data(self, df_real_train: pd.DataFrame):
        """Fits empirical models strictly on historical development data (2020-2021)."""
        # 1. Enforce Locked Evaluation Firewall
        self.firewall.verify_training_partition_isolation(df_real_train, "historical_training_dataset")

        logger.info(f"Fitting HP-STG models on authorized training partition (N={len(df_real_train)})...")
        self.hp_stg_generator.fit_from_training_data(df_real_train)

        # Fit OOD and Memorization baselines
        self.ood_scorer.fit(df_real_train)
        self.memorization_screen.fit(df_real_train)

        self._is_fit = True
        logger.info("HP-STG v1.0.0 successfully fit and calibrated.")

    def generate_single_trajectory(
        self,
        trajectory_id: str,
        length: int,
        season: str,
        trajectory_seed: int
    ) -> pd.DataFrame:
        """Generates a single continuous synthetic trajectory using deterministic seed."""
        # Set generator RNG to deterministic trajectory seed
        self.hp_stg_generator.rng = np.random.RandomState(trajectory_seed)
        self.hp_stg_generator.regime_model.rng = self.hp_stg_generator.rng
        self.hp_stg_generator.seasonal_model.rng = self.hp_stg_generator.rng
        self.hp_stg_generator.physics_model.rng = self.hp_stg_generator.rng
        self.hp_stg_generator.innovation_sampler.rng = self.hp_stg_generator.rng

        df_traj = self.hp_stg_generator.generate_single_trajectory(
            trajectory_id=trajectory_id,
            length=length,
            season=season,
        )

        gen_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Attach production Phase 8A metadata
        df_traj["trajectory_id"] = trajectory_id
        df_traj["data_origin"] = "synthetic"
        df_traj["generator_version"] = self.config.generator_version
        df_traj["phase_version"] = self.config.phase_version
        df_traj["source_partition"] = self.config.source_partition_name
        df_traj["generation_seed"] = trajectory_seed
        df_traj["global_seed"] = self.config.global_seed
        df_traj["synthetic_timestamp"] = df_traj["synthetic_date"]
        df_traj["generation_timestamp"] = gen_ts

        # Annotate with OOD feature space metrics
        df_annotated = self.ood_scorer.annotate_trajectory(df_traj)

        return df_annotated

    def generate_batch(
        self,
        trajectory_specs: List[Tuple[int, str]]
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], Dict[str, Any]]:
        """
        Generates a batch of trajectories, performing physics validation,
        extreme filtering, and memorization checks.
        """
        if not self._is_fit:
            raise RuntimeError("Generator must be fit on training data before batch generation.")

        accepted = []
        rejected = []
        gen_version_clean = self.config.generator_version.replace(".", "_")

        for idx, (length, season) in enumerate(trajectory_specs):
            traj_id = f"SYNTH_8A_{gen_version_clean}_{self.config.global_seed}_{idx+1:05d}"
            traj_seed = Phase8AProvenanceManager.derive_trajectory_seed(self.config.global_seed, traj_id)

            df_traj = self.generate_single_trajectory(traj_id, length, season, traj_seed)

            # 1. Physics Validation
            phys_ok, phys_report = self.physics_validator.validate_trajectory(df_traj, traj_id)

            # 2. Extreme-Tail Environmental Filtering (Restriction C)
            ext_ok, ext_rejections = self.extreme_filter.evaluate_trajectory(
                df_traj, traj_id, traj_seed, self.config.generator_version, self.config.phase_version
            )

            # 3. Memorization Screen
            mem_ok, mem_report = self.memorization_screen.screen_trajectory(df_traj, traj_id)

            # Decision
            if phys_ok and ext_ok and mem_ok:
                accepted.append(df_traj)
            else:
                rejected.append(df_traj)
                logger.warning(f"Trajectory {traj_id} REJECTED: phys_ok={phys_ok}, ext_ok={ext_ok}, mem_ok={mem_ok}")

        stats = {
            "requested_trajectories": len(trajectory_specs),
            "accepted_trajectories": len(accepted),
            "rejected_trajectories": len(rejected),
            "accepted_observations": sum(len(df) for df in accepted),
            "rejected_observations": sum(len(df) for df in rejected),
            "acceptance_rate_pct": float(len(accepted) / len(trajectory_specs) * 100.0) if len(trajectory_specs) > 0 else 0.0,
        }

        return accepted, rejected, stats
