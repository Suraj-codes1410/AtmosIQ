"""
AtmosIQ Phase 7C: Phase 6F Production Freeze Gate Verification Engine.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple


class Phase6FFreezeVerifier:
    """Verifies all 21 protected Phase 6F production and dataset artifacts."""

    def __init__(self, root_dir: Path, freeze_manifest_path: Path):
        self.root_dir = Path(root_dir)
        self.freeze_manifest_path = Path(freeze_manifest_path)

    def verify_freeze_baseline(self) -> Tuple[bool, Dict[str, Any]]:
        if not self.freeze_manifest_path.exists():
            raise FileNotFoundError(f"Phase 6F freeze manifest missing at {self.freeze_manifest_path}")

        with open(self.freeze_manifest_path) as f:
            freeze_data = json.load(f)

        artifact_results = {}
        all_passed = True

        for rel_path, exp_hash in freeze_data["artifact_hashes"].items():
            p = self.root_dir / rel_path
            if not p.exists():
                artifact_results[rel_path] = {
                    "expected_sha256": exp_hash,
                    "actual_sha256": "MISSING",
                    "status": "FAIL_MISSING",
                }
                all_passed = False
                continue

            act_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            is_match = (act_hash == exp_hash)
            if not is_match:
                all_passed = False

            artifact_results[rel_path] = {
                "expected_sha256": exp_hash,
                "actual_sha256": act_hash,
                "status": "PASS" if is_match else "FAIL_HASH_MISMATCH",
            }

        verification_record = {
            "phase": "Phase 7C",
            "freeze_gate_status": "PASS" if all_passed else "FAIL",
            "total_protected_artifacts": len(freeze_data["artifact_hashes"]),
            "artifacts": artifact_results,
        }

        return all_passed, verification_record
