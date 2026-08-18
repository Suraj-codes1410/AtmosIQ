"""
AtmosIQ Phase 8G: Integration Manifest & Training Provenance Manager.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import hashlib
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Phase8GManifestManager:
    """Manages the generation and cryptographic serialization of Phase 8G manifests."""

    def __init__(self, manifests_dir: Path):
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def generate_integration_manifest(
        self,
        config_dict: Dict[str, Any],
        cal_07_sha: str,
        prod_sha: str,
        contract_sha: str,
        integration_matrix: List[Dict[str, Any]],
        audits_summary: Dict[str, str]
    ) -> Dict[str, Any]:
        """Creates the formal Phase 8G integration manifest."""
        manifest = {
            "manifest_name": "AtmosIQ_Phase8G_Production_Integration_Manifest",
            "manifest_version": "v1.0.0",
            "integration_status": "APPROVED_FOR_PHASE_9",
            "production_synthetic_corpus": {
                "name": "AtmosIQ_Synthetic_Production",
                "version": "v1.0.0",
                "sha256": prod_sha,
                "role": "CANONICAL_PRODUCTION_BENCHMARK",
            },
            "preferred_research_corpus": {
                "name": "AtmosIQ_Synthetic_Calibrated",
                "version": "v0.1.0",
                "candidate": "CAL-07",
                "sha256": cal_07_sha,
                "role": "PREFERRED_PHASE_9_TRAINING_CORPUS",
            },
            "phase9_training_contract": {
                "contract_file": "phase9_training_contract.json",
                "contract_sha256": contract_sha,
                "status": "VALIDATED_AND_ENFORCED",
            },
            "governance_augmentation_rules": {
                "recommended_production_ratio": 0.25,
                "controlled_upper_bound_ratio": 0.50,
                "prohibited_ratio": 1.00,
            },
            "integration_configurations": integration_matrix,
            "audits_summary": audits_summary,
            "scientific_safeguards": [
                "SYNTHETIC DATA != OBSERVED DATA",
                "PHYSICS-INFORMED != PHYSICALLY EXACT",
                "STATISTICAL FIDELITY != CAUSAL VALIDATION",
                "ML UTILITY != SCIENTIFIC TRUTH",
                "SYNTHETIC AUGMENTATION != REAL-WORLD OBSERVATION",
            ],
        }

        # Calculate manifest SHA-256
        encoded = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest_sha = hashlib.sha256(encoded).hexdigest()
        manifest["manifest_sha256"] = manifest_sha

        manifest_path = self.manifests_dir / "phase8g_integration_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

        logger.info(f"Phase 8G Integration Manifest written to {manifest_path} (SHA: {manifest_sha[:16]}...).")
        return manifest
