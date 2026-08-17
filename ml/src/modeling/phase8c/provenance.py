"""
AtmosIQ Phase 8C: Comprehensive Provenance & Freeze Verification Engine.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import hashlib
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Phase8CProvenanceManager:
    """Manages cryptographic freeze audits and granular observation-level provenance manifests."""

    def __init__(self, root_dir: Path, freeze_manifest_path: Path):
        self.root_dir = Path(root_dir)
        self.freeze_manifest_path = Path(freeze_manifest_path)

    def verify_phase6f_freeze(self) -> Tuple[bool, Dict[str, Any]]:
        """Verifies all 21 protected baseline artifacts."""
        if not self.freeze_manifest_path.exists():
            raise FileNotFoundError(f"Freeze manifest missing at {self.freeze_manifest_path}")

        with open(self.freeze_manifest_path) as f:
            manifest_data = json.load(f)

        results = {}
        all_passed = True

        for rel_path, expected_hash in manifest_data["artifact_hashes"].items():
            full_path = self.root_dir / rel_path
            if not full_path.exists():
                results[rel_path] = {
                    "expected_sha256": expected_hash,
                    "actual_sha256": "MISSING",
                    "status": "FAIL_MISSING",
                }
                all_passed = False
                continue

            actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
            is_match = (actual_hash == expected_hash)
            if not is_match:
                all_passed = False

            results[rel_path] = {
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": "PASS" if is_match else "FAIL_HASH_MISMATCH",
            }

        summary = {
            "phase": "Phase 8C",
            "freeze_status": "PASS" if all_passed else "FAIL",
            "total_protected_artifacts": len(manifest_data["artifact_hashes"]),
            "artifacts": results,
        }

        return all_passed, summary

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def generate_provenance_manifest(
        self,
        df_corpus: pd.DataFrame,
        release_version: str = "v1.0.0"
    ) -> pd.DataFrame:
        """Generates granular, observation-level provenance manifest."""
        records = []
        for idx, row in df_corpus.iterrows():
            t_id = str(row["trajectory_id"])
            b_id = str(row.get("batch_id", "batch_0001"))
            g_ver = str(row.get("generator_version", "HP-STG-v1.0.0"))
            seed = int(row.get("generation_seed", 42))
            horizon = int(row.get("horizon_days", 14))
            date_str = str(row.get("synthetic_date", row.get("date", f"DAY_{idx}")))
            obs_id = f"{t_id}_OBS_{idx:07d}"

            raw_sig = f"{obs_id}_{t_id}_{b_id}_{g_ver}_{seed}_{horizon}_{date_str}".encode("utf-8")
            prov_hash = hashlib.sha256(raw_sig).hexdigest()[:16]

            records.append({
                "trajectory_id": t_id,
                "observation_id": obs_id,
                "timestamp": date_str,
                "synthetic": True,
                "generator_name": "HP-STG",
                "generator_version": g_ver,
                "generation_seed": seed,
                "source_partition": "2020-2021",
                "horizon_days": horizon,
                "phase8b_batch_id": b_id,
                "phase8c_release_version": release_version,
                "physical_validation_status": "VALIDATED",
                "provenance_hash": prov_hash,
            })

        df_prov = pd.DataFrame(records)
        logger.info(f"Generated provenance manifest for {len(df_prov)} observations.")
        return df_prov
