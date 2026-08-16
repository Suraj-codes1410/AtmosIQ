"""
AtmosIQ Phase 8A: Machine-Readable Dataset Manifest & Provenance Generator.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DatasetManifestGenerator:
    """Generates machine-readable manifest and audit records for the synthetic dataset."""

    def __init__(self, manifests_dir: Path):
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def generate_manifest(
        self,
        config_dict: Dict[str, Any],
        source_dataset_sha256: str,
        shard_records: List[Dict[str, Any]],
        generation_stats: Dict[str, Any],
        df_rejections: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generates comprehensive dataset_manifest.json and writes rejection audit."""
        manifest_data = {
            "dataset_version": config_dict["dataset_version"],
            "generator_version": config_dict["generator_version"],
            "phase_version": config_dict["phase_version"],
            "generation_mode": config_dict["mode"],
            "source_partition": config_dict["source_partition"],
            "source_dataset_sha256": source_dataset_sha256,
            "configuration_sha256": config_dict["config_sha256"],
            "global_seed": config_dict["global_seed"],
            "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "trajectory_count": generation_stats["accepted_trajectories"],
            "observation_count": generation_stats["accepted_observations"],
            "trajectory_lengths_supported": config_dict["trajectory_lengths"],
            "augmentation_ratio": config_dict["augmentation_ratio"],
            "accepted_trajectory_count": generation_stats["accepted_trajectories"],
            "rejected_trajectory_count": generation_stats["rejected_trajectories"],
            "acceptance_rate_pct": generation_stats["acceptance_rate_pct"],
            "shards": shard_records,
            "rejection_statistics": {
                "total_rejections": len(df_rejections),
                "rejection_reasons": df_rejections["rejection_reason"].value_counts().to_dict() if len(df_rejections) > 0 else {},
            },
        }

        # Write dataset_manifest.json
        manifest_file = self.manifests_dir / "dataset_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f, indent=4)
        logger.info(f"Dataset manifest written to {manifest_file}")

        # Write rejection_audit.csv
        rejection_file = self.manifests_dir / "rejection_audit.csv"
        df_rejections.to_csv(rejection_file, index=False)
        logger.info(f"Rejection audit written to {rejection_file} ({len(df_rejections)} rejections)")

        # Write provenance.json
        prov_data = {
            "dataset_version": config_dict["dataset_version"],
            "source_data_provenance": {
                "partition": config_dict["source_partition"],
                "sha256": source_dataset_sha256,
            },
            "generator_provenance": {
                "version": config_dict["generator_version"],
                "seed": config_dict["global_seed"],
                "config_sha256": config_dict["config_sha256"],
            },
            "shards_provenance": [
                {"shard_name": s["shard_name"], "sha256": s["sha256"], "rows": s["observation_count"]}
                for s in shard_records
            ]
        }
        prov_file = self.manifests_dir / "provenance.json"
        with open(prov_file, "w") as f:
            json.dump(prov_data, f, indent=4)

        return manifest_data
