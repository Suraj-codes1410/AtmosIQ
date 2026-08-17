"""
AtmosIQ Phase 8B: Scalable Batch Trajectory Generation Engine.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import logging

from ml.src.modeling.phase7b.config import SyntheticConfigPhase7B
from ml.src.modeling.phase7b.trajectory_generator import TrajectoryGeneratorPhase7B

from .config import ScalingConfigPhase8B
from .provenance import Phase8BProvenanceManager
from .validation import Phase8BPhysicsValidator
from .ood_monitor import OODScaleMonitor
from .memorization import MemorizationScaleAuditor
from ml.src.modeling.phase8a.filtering import ExtremeTailFilter
from ml.src.modeling.phase8a.firewall import EvaluationFirewall

logger = logging.getLogger(__name__)


class ScalingBatchGenerator:
    """Generates stratified, validated synthetic trajectory batches with checkpointing."""

    def __init__(self, config: ScalingConfigPhase8B, feature_registry: List[str]):
        self.config = config
        self.feature_registry = list(feature_registry)

        synth_cfg = SyntheticConfigPhase7B(
            random_seed=self.config.global_master_seed,
            training_start_date=self.config.dev_train_start_date,
            training_end_date=self.config.dev_train_end_date,
            locked_test_start_date=self.config.locked_eval_start_date,
        )
        self.hp_stg_generator = TrajectoryGeneratorPhase7B(synth_cfg, self.feature_registry)

        self.firewall = EvaluationFirewall(self.config.locked_eval_start_date)
        self.physics_validator = Phase8BPhysicsValidator()
        self.extreme_filter = ExtremeTailFilter(
            enabled=self.config.extreme_filter_enabled,
            extreme_pm25_threshold=self.config.extreme_pm25_threshold,
            vi_threshold=self.config.vi_threshold,
            precipitation_threshold=self.config.precipitation_threshold,
        )
        self.ood_monitor = OODScaleMonitor(self.feature_registry)
        self.memorization_auditor = MemorizationScaleAuditor(self.feature_registry)

        self._is_fit = False

    def fit(self, df_real_dev: pd.DataFrame):
        """Fits empirical models strictly on historical development data (2020-2021)."""
        self.firewall.verify_training_partition_isolation(df_real_dev, "historical_training_dataset")

        self.hp_stg_generator.fit_from_training_data(df_real_dev)
        self.ood_monitor.fit(df_real_dev)
        self.memorization_auditor.fit(df_real_dev)

        self._is_fit = True
        logger.info(f"ScalingBatchGenerator fitted successfully on {len(df_real_dev)} development rows.")

    def generate_batch(
        self,
        batch_id: str,
        target_trajectories: int,
        batch_dir: Path
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """
        Generates an individual scaling batch with stratified seasons and horizons.
        Saves batch parquet shards and returns (df_accepted, batch_metadata, df_rejections).
        """
        if not self._is_fit:
            raise RuntimeError("Generator must be fit on development data before generation.")

        batch_dir = Path(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_seed = Phase8BProvenanceManager.derive_batch_seed(self.config.global_master_seed, batch_id)
        seasons_cycle = ["Winter", "Post-Monsoon", "Summer", "Monsoon"]

        accepted_trajs: List[pd.DataFrame] = []
        rejected_trajs: List[pd.DataFrame] = []

        local_filter = ExtremeTailFilter(
            enabled=self.config.extreme_filter_enabled,
            extreme_pm25_threshold=self.config.extreme_pm25_threshold,
            vi_threshold=self.config.vi_threshold,
            precipitation_threshold=self.config.precipitation_threshold,
        )

        logger.info(f"Starting generation for {batch_id}: target = {target_trajectories} trajectories...")

        for idx in range(target_trajectories):
            traj_id = f"SYNTH_8B_{batch_id}_{idx+1:05d}"
            traj_seed = Phase8BProvenanceManager.derive_trajectory_seed(batch_seed, traj_id)
            horizon = 14 if (idx % 2 == 0) else 30
            season = seasons_cycle[idx % len(seasons_cycle)]

            # Seed underlying generator
            self.hp_stg_generator.rng = np.random.RandomState(traj_seed)
            self.hp_stg_generator.regime_model.rng = self.hp_stg_generator.rng
            self.hp_stg_generator.seasonal_model.rng = self.hp_stg_generator.rng
            self.hp_stg_generator.physics_model.rng = self.hp_stg_generator.rng
            self.hp_stg_generator.innovation_sampler.rng = self.hp_stg_generator.rng

            df_traj = self.hp_stg_generator.generate_single_trajectory(
                trajectory_id=traj_id,
                length=horizon,
                season=season,
            )

            # Metadata
            gen_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            df_traj["trajectory_id"] = traj_id
            df_traj["batch_id"] = batch_id
            df_traj["data_origin"] = "synthetic"
            df_traj["generator_version"] = self.config.generator_version
            df_traj["phase_version"] = "Phase 8B v1.0.0"
            df_traj["source_partition"] = self.config.source_partition_name
            df_traj["generation_seed"] = traj_seed
            df_traj["batch_seed"] = batch_seed
            df_traj["global_master_seed"] = self.config.global_master_seed
            df_traj["horizon_days"] = horizon
            df_traj["generation_timestamp"] = gen_ts

            # 1. Physics Validation
            phys_ok, _ = self.physics_validator.validate_trajectory(df_traj, traj_id)

            # 2. Extreme-Tail Filter
            ext_ok, _ = local_filter.evaluate_trajectory(
                df_traj, traj_id, traj_seed, self.config.generator_version, "Phase 8B v1.0.0"
            )

            # 3. Memorization Screen
            mem_report = self.memorization_auditor.audit_batch(df_traj, batch_id)
            mem_ok = (mem_report["memorization_status"] == "PASS")

            if phys_ok and ext_ok and mem_ok:
                accepted_trajs.append(df_traj)
            else:
                rejected_trajs.append(df_traj)

        # Consolidate Accepted
        if accepted_trajs:
            df_accepted_batch = pd.concat(accepted_trajs, ignore_index=True)
            # Annotate OOD
            ood_sum, df_annotated = self.ood_monitor.evaluate_batch_ood(df_accepted_batch, batch_id)
            df_final = df_annotated
        else:
            df_final = pd.DataFrame()
            ood_sum = {}

        # Save Shards
        shard_path = batch_dir / f"{batch_id}_accepted.parquet"
        if len(df_final) > 0:
            df_final.to_parquet(shard_path, index=False)
            shard_sha = Phase8BProvenanceManager.compute_file_sha256(shard_path)
        else:
            shard_sha = "EMPTY"

        df_rejections = local_filter.get_rejection_dataframe()
        df_rejections.to_csv(batch_dir / f"{batch_id}_rejections.csv", index=False)

        batch_meta = {
            "batch_id": batch_id,
            "target_trajectories": target_trajectories,
            "accepted_trajectories": len(accepted_trajs),
            "rejected_trajectories": len(rejected_trajs),
            "accepted_observations": len(df_final),
            "acceptance_rate_pct": float(len(accepted_trajs) / target_trajectories * 100.0) if target_trajectories > 0 else 0.0,
            "shard_path": str(shard_path),
            "shard_sha256": shard_sha,
            "batch_seed": batch_seed,
            "ood_summary": ood_sum,
        }

        logger.info(
            f"Batch {batch_id} complete: {len(accepted_trajs)}/{target_trajectories} accepted "
            f"({len(df_final)} rows, Acceptance: {batch_meta['acceptance_rate_pct']:.1f}%)."
        )

        return df_final, batch_meta, df_rejections
