"""
AtmosIQ Phase 8B: Hierarchical Provenance Manager & Deterministic Seed Engine.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class Phase8BProvenanceManager:
    """Manages Phase 6F freeze verification and hierarchical deterministic seed derivation."""

    def __init__(self, root_dir: Path, freeze_manifest_path: Path):
        self.root_dir = Path(root_dir)
        self.freeze_manifest_path = Path(freeze_manifest_path)

    def verify_phase6f_freeze(self) -> Tuple[bool, Dict[str, Any]]:
        """Verifies all 21 protected Phase 6F baseline artifacts."""
        if not self.freeze_manifest_path.exists():
            raise FileNotFoundError(f"Phase 6F freeze manifest missing at {self.freeze_manifest_path}")

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
            "phase": "Phase 8B",
            "freeze_status": "PASS" if all_passed else "FAIL",
            "total_protected_artifacts": len(manifest_data["artifact_hashes"]),
            "artifacts": results,
        }

        return all_passed, summary

    @staticmethod
    def derive_batch_seed(master_seed: int, batch_id: str) -> int:
        """Derives deterministic batch seed from master seed and batch ID."""
        raw = f"{master_seed}_{batch_id}".encode("utf-8")
        hex_digest = hashlib.sha256(raw).hexdigest()[:8]
        return int(hex_digest, 16)

    @staticmethod
    def derive_trajectory_seed(batch_seed: int, trajectory_id: str) -> int:
        """Derives deterministic trajectory seed from batch seed and trajectory ID."""
        raw = f"{batch_seed}_{trajectory_id}".encode("utf-8")
        hex_digest = hashlib.sha256(raw).hexdigest()[:8]
        return int(hex_digest, 16)

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def generate_checksums_file(self, target_files: List[Path], output_path: Path):
        """Generates checksums.txt file."""
        lines = []
        for p in sorted(target_files):
            p = Path(p)
            if p.exists() and p.is_file():
                h = self.compute_file_sha256(p)
                lines.append(f"{h}  {p.name}\n")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(lines)
        logger.info(f"Generated SHA-256 checksums at {output_path} ({len(lines)} files).")
