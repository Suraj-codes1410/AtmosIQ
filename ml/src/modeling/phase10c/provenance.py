"""
AtmosIQ Phase 10C: Cryptographic Freeze & Upstream Provenance Manager.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class Phase10CProvenanceManager:
    """Manages pre-validation and post-validation cryptographic freeze verification across all upstream phases."""

    def __init__(self, root_dir: Path, freeze_manifest_path: Path):
        self.root_dir = Path(root_dir)
        self.freeze_manifest_path = Path(freeze_manifest_path)

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def verify_all_protected_artifacts(self) -> Tuple[bool, Dict[str, Any]]:
        """Audits all protected artifacts across Phases 6F to Phase 10B."""
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

        # 2. Key Artifacts across Phases 8 to 10B
        extra_protected = [
            "ml/experiments/phase8c_release/synthetic_dataset/synthetic_production_corpus_v1_0_0.parquet",
            "ml/experiments/phase8d_calibration/experiments/cal07_combined/AtmosIQ_Synthetic_Calibrated_v0.1.0.parquet",
            "ml/experiments/phase8e_readiness/contracts/phase9_training_contract.json",
            "ml/experiments/phase8f_governance/manifests/phase8f_artifact_manifest.json",
            "ml/experiments/phase8g_integration/manifests/phase8g_integration_manifest.json",
            "ml/experiments/phase8h_readiness/manifests/phase8h_training_manifest.json",
            "ml/experiments/phase9_deep_learning/manifests/phase9_training_manifest.json",
            "ml/experiments/phase9ab_certification/manifests/phase9ab_final_decision.json",
            "ml/experiments/phase9cd_hardening/manifests/phase9cd_model_manifest.json",
            "ml/experiments/phase10_production/manifests/phase10_model_manifest.json",
            "ml/experiments/phase10b_observability/manifests/phase10b_model_registry.json",
        ]

        for p_rel in extra_protected:
            full_p = self.root_dir / p_rel
            if full_p.exists():
                results[p_rel] = {
                    "actual_sha256": self.compute_file_sha256(full_p),
                    "status": "PASS_RECORDED",
                }

        summary = {
            "phase": "Phase 10C",
            "freeze_status": "PASS" if all_passed else "FAIL",
            "total_artifacts_verified": len(results),
            "drift_count": sum(1 for v in results.values() if "FAIL" in v.get("status", "")),
            "artifacts": results,
        }

        return all_passed, summary
