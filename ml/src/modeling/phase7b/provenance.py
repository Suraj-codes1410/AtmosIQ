"""
AtmosIQ Phase 7B: Provenance and Freeze Gate Verification.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ProvenanceVerifierPhase7B:
    PHASE_7A_EXPECTED_HASH = "813982d09e0cb8c7ec5151d8e0979729f47ef318ef5a765fdbb57d239072b694"

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.freeze_manifest_path = self.root_dir / "ml/experiments/phase6f/phase6f_freeze_manifest.json"
        self.phase7a_spec_path = self.root_dir / "docs/phase7/phase7a_physics_informed_synthetic_data_spec.md"

    def verify_phase7a_spec(self) -> Tuple[bool, str]:
        if not self.phase7a_spec_path.exists():
            raise FileNotFoundError(f"Phase 7A specification missing at {self.phase7a_spec_path}")
        actual_hash = hashlib.sha256(self.phase7a_spec_path.read_bytes()).hexdigest()
        passed = (actual_hash == self.PHASE_7A_EXPECTED_HASH)
        if not passed:
            logger.warning(f"Phase 7A spec hash mismatch! Expected {self.PHASE_7A_EXPECTED_HASH}, got {actual_hash}")
        return passed, actual_hash

    def verify_phase6f_freeze(self) -> Tuple[bool, Dict[str, str]]:
        if not self.freeze_manifest_path.exists():
            raise FileNotFoundError(f"Phase 6F freeze manifest missing at {self.freeze_manifest_path}")

        with open(self.freeze_manifest_path) as f:
            freeze_data = json.load(f)

        mismatches = {}
        for rel_path, exp_hash in freeze_data["artifact_hashes"].items():
            p = self.root_dir / rel_path
            if not p.exists():
                mismatches[rel_path] = "MISSING_FILE"
                continue
            act_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            if act_hash != exp_hash:
                mismatches[rel_path] = f"HASH_MISMATCH: expected {exp_hash[:8]}, got {act_hash[:8]}"

        all_passed = (len(mismatches) == 0)
        return all_passed, mismatches

    def get_provenance_summary(self) -> Dict[str, Any]:
        spec_pass, spec_hash = self.verify_phase7a_spec()
        freeze_pass, mismatches = self.verify_phase6f_freeze()
        return {
            "phase7a_spec_verified": spec_pass,
            "phase7a_spec_sha256": spec_hash,
            "phase6f_freeze_verified": freeze_pass,
            "freeze_violations": mismatches,
        }
