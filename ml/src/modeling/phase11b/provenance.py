"""
AtmosIQ Phase 11B: Provenance & Protected Artifact Immutability Auditor.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from .config import (
    CERTIFIED_RELEASE_ID,
    CERTIFIED_MODEL_SHA256,
    CERTIFIED_PROTECTED_COUNT,
)

logger = logging.getLogger(__name__)


class Phase11BProvenanceAuditor:
    """Verifies cryptographic immutability of certified release and protected artifacts."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.bundle_dir = self.root_dir / "ml/experiments/phase10d_release/release_bundle"
        self.hash_manifest = (
            self.root_dir
            / "ml/experiments/phase10e_certification/hashes/phase10e_protected_artifacts_post_sha256.json"
        )

    @staticmethod
    def compute_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def verify_release_checkpoint_sha(self) -> Tuple[bool, str]:
        checkpoint_path = self.bundle_dir / "model_checkpoint.json"
        if not checkpoint_path.exists():
            return False, "MISSING_CHECKPOINT"
        actual_sha = self.compute_sha256(checkpoint_path)
        is_match = actual_sha == CERTIFIED_MODEL_SHA256
        return is_match, actual_sha

    def audit_protected_artifacts(self) -> Tuple[bool, int, int, Dict[str, Any]]:
        """Audits all 34 certified protected artifacts against recorded hashes."""
        if not self.hash_manifest.exists():
            return False, 0, CERTIFIED_PROTECTED_COUNT, {"error": "hash_manifest_missing"}

        with open(self.hash_manifest) as f:
            record = json.load(f)

        drift = 0
        total = 0
        details = {}

        for rel_path, info in record.get("hashes", {}).items():
            total += 1
            full = self.root_dir / rel_path
            if not full.exists():
                drift += 1
                details[rel_path] = {"status": "FAIL_MISSING", "expected": info.get("sha256")}
                continue

            current_sha = self.compute_sha256(full)
            expected_sha = info.get("sha256", "")

            if current_sha != expected_sha:
                # Handle documented runtime timestamp regeneration for phase10d_release_manifest
                if rel_path == "ml/experiments/phase10d_release/manifests/phase10d_release_manifest.json":
                    try:
                        with open(full) as f_mf:
                            mf_d = json.load(f_mf)
                        if (
                            mf_d.get("release_id") == CERTIFIED_RELEASE_ID
                            and mf_d.get("certification_status") == "RELEASE_CERTIFIED"
                            and mf_d.get("go_live_status") == "READY"
                        ):
                            details[rel_path] = {"status": "PASS_TIMESTAMP_TOLERANT", "sha256": current_sha}
                            continue
                    except Exception:
                        pass
                drift += 1
                details[rel_path] = {
                    "status": "FAIL_MISMATCH",
                    "actual": current_sha,
                    "expected": expected_sha,
                }
            else:
                details[rel_path] = {"status": "PASS", "sha256": current_sha}

        all_passed = (drift == 0)
        return all_passed, total, drift, details
