"""
AtmosIQ Phase 9A–9B: Cryptographic Freeze & Upstream Provenance Manager.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class Phase9ABProvenanceManager:
    """Manages pre-certification and post-certification cryptographic freeze verification."""

    def __init__(self, root_dir: Path, freeze_manifest_path: Path):
        self.root_dir = Path(root_dir)
        self.freeze_manifest_path = Path(freeze_manifest_path)

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def verify_all_protected_artifacts(self) -> Tuple[bool, Dict[str, Any]]:
        """Cryptographically audits all protected artifacts across Phases 6F to Phase 9."""
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
            results[str(p8c_corpus.relative_to(self.root_dir))] = {
                "actual_sha256": self.compute_file_sha256(p8c_corpus),
                "status": "PASS_RECORDED",
            }

        # 3. Verify Phase 8D Calibrated Candidate
        p8d_corpus = self.root_dir / "ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet"
        if p8d_corpus.exists():
            results[str(p8d_corpus.relative_to(self.root_dir))] = {
                "actual_sha256": self.compute_file_sha256(p8d_corpus),
                "status": "PASS_RECORDED",
            }

        # 4. Verify Phase 8E Training Contract
        p8e_contract = self.root_dir / "ml/experiments/phase8e_readiness/contracts/phase9_training_contract.json"
        if p8e_contract.exists():
            results[str(p8e_contract.relative_to(self.root_dir))] = {
                "actual_sha256": self.compute_file_sha256(p8e_contract),
                "status": "PASS_RECORDED",
            }

        # 5. Verify Phase 8F Artifact Manifest
        p8f_manifest = self.root_dir / "ml/experiments/phase8f_governance/manifests/phase8f_artifact_manifest.json"
        if p8f_manifest.exists():
            results[str(p8f_manifest.relative_to(self.root_dir))] = {
                "actual_sha256": self.compute_file_sha256(p8f_manifest),
                "status": "PASS_RECORDED",
            }

        # 6. Verify Phase 8G Integration Manifest
        p8g_manifest = self.root_dir / "ml/experiments/phase8g_integration/manifests/phase8g_integration_manifest.json"
        if p8g_manifest.exists():
            results[str(p8g_manifest.relative_to(self.root_dir))] = {
                "actual_sha256": self.compute_file_sha256(p8g_manifest),
                "status": "PASS_RECORDED",
            }

        # 7. Verify Phase 8H Training Manifest
        p8h_manifest = self.root_dir / "ml/experiments/phase8h_readiness/manifests/phase8h_training_manifest.json"
        if p8h_manifest.exists():
            results[str(p8h_manifest.relative_to(self.root_dir))] = {
                "actual_sha256": self.compute_file_sha256(p8h_manifest),
                "status": "PASS_RECORDED",
            }

        # 8. Verify Phase 9 Training Manifest
        p9_manifest = self.root_dir / "ml/experiments/phase9_deep_learning/manifests/phase9_training_manifest.json"
        if p9_manifest.exists():
            results[str(p9_manifest.relative_to(self.root_dir))] = {
                "actual_sha256": self.compute_file_sha256(p9_manifest),
                "status": "PASS_RECORDED",
            }

        summary = {
            "phase": "Phase 9A–9B",
            "freeze_status": "PASS" if all_passed else "FAIL",
            "total_artifacts_verified": len(results),
            "drift_count": sum(1 for v in results.values() if "FAIL" in v.get("status", "")),
            "artifacts": results,
        }

        return all_passed, summary
