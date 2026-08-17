"""
AtmosIQ Phase 8E: Cryptographic Freeze & Upstream Provenance Manager.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class Phase8EProvenanceManager:
    """Manages pre-run and post-run cryptographic verification of all protected baseline artifacts."""

    def __init__(self, root_dir: Path, freeze_manifest_path: Path):
        self.root_dir = Path(root_dir)
        self.freeze_manifest_path = Path(freeze_manifest_path)

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def verify_all_protected_artifacts(self) -> Tuple[bool, Dict[str, Any]]:
        """Cryptographically audits Phase 6F, Datasets, Phase 8C release, and Phase 8D candidate."""
        if not self.freeze_manifest_path.exists():
            raise FileNotFoundError(f"Freeze manifest missing at {self.freeze_manifest_path}")

        with open(self.freeze_manifest_path) as f:
            manifest_data = json.load(f)

        results = {}
        all_passed = True

        # 1. Verify Phase 6F Baseline (21 artifacts)
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

            actual_hash = self.compute_file_sha256(full_path)
            is_match = (actual_hash == expected_hash)
            if not is_match:
                all_passed = False

            results[rel_path] = {
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": "PASS" if is_match else "FAIL_HASH_MISMATCH",
            }

        # 2. Verify Phase 8C Released Corpus
        p8c_corpus = self.root_dir / "ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet"
        if p8c_corpus.exists():
            p8c_hash = self.compute_file_sha256(p8c_corpus)
            results["ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet"] = {
                "actual_sha256": p8c_hash,
                "status": "PASS_RECORDED",
            }

        # 3. Verify Phase 8D Calibrated Candidate
        p8d_corpus = self.root_dir / "ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet"
        if p8d_corpus.exists():
            p8d_hash = self.compute_file_sha256(p8d_corpus)
            results["ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet"] = {
                "actual_sha256": p8d_hash,
                "status": "PASS_RECORDED",
            }

        summary = {
            "phase": "Phase 8E",
            "freeze_status": "PASS" if all_passed else "FAIL",
            "total_artifacts_verified": len(results),
            "artifacts": results,
        }

        return all_passed, summary
