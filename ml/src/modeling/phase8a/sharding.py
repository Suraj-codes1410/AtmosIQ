"""
AtmosIQ Phase 8A: Dataset Sharding Engine.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import hashlib
import logging

logger = logging.getLogger(__name__)


class DatasetSharder:
    """Partitions accepted synthetic trajectories into deterministic Parquet shards."""

    def __init__(
        self,
        shards_dir: Path,
        max_trajectories_per_shard: int = 10,
        config_hash: str = ""
    ):
        self.shards_dir = Path(shards_dir)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.max_trajectories_per_shard = max_trajectories_per_shard
        self.config_hash = config_hash

    def write_shards(self, accepted_trajectories: List[pd.DataFrame]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Groups trajectories into shards and writes parquet files.
        Returns (shard_metadata_list, consolidated_dataframe).
        """
        if not accepted_trajectories:
            return [], pd.DataFrame()

        # Clean existing shards
        for p in self.shards_dir.glob("shard-*.parquet"):
            p.unlink()

        shard_records = []
        consolidated_dfs = []
        total_trajs = len(accepted_trajectories)

        num_shards = (total_trajs + self.max_trajectories_per_shard - 1) // self.max_trajectories_per_shard

        for s_idx in range(num_shards):
            start_i = s_idx * self.max_trajectories_per_shard
            end_i = min(start_i + self.max_trajectories_per_shard, total_trajs)

            shard_trajs = accepted_trajectories[start_i:end_i]
            df_shard = pd.concat(shard_trajs, ignore_index=True)
            consolidated_dfs.append(df_shard)

            shard_filename = f"shard-{s_idx+1:06d}.parquet"
            shard_path = self.shards_dir / shard_filename

            df_shard.to_parquet(shard_path, index=False)
            shard_sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()

            rec = {
                "shard_name": shard_filename,
                "shard_index": s_idx + 1,
                "shard_path": str(shard_path),
                "trajectory_count": len(shard_trajs),
                "observation_count": len(df_shard),
                "feature_count": len(df_shard.columns),
                "sha256": shard_sha256,
                "config_hash": self.config_hash,
            }
            shard_records.append(rec)
            logger.info(f"Wrote {shard_filename}: {len(df_shard)} rows, {len(shard_trajs)} trajectories, SHA: {shard_sha256[:12]}...")

        df_consolidated = pd.concat(consolidated_dfs, ignore_index=True)
        return shard_records, df_consolidated
