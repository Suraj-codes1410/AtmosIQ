"""
AtmosIQ Phase 9C–9D: Manifest Management & Candidate Decision Serialization.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Phase9CDManifestManager:
    """Manages serialization of model manifests, preprocessing manifests, and candidate comparison decisions."""

    def __init__(self, manifests_dir: Path, benchmarks_dir: Path):
        self.manifests_dir = Path(manifests_dir)
        self.benchmarks_dir = Path(benchmarks_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

    def export_model_manifest(self, data: Dict[str, Any]) -> Path:
        p = self.manifests_dir / "phase9cd_model_manifest.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=4)
        return p

    def export_preprocessing_manifest(self, data: Dict[str, Any]) -> Path:
        p = self.manifests_dir / "phase9cd_preprocessing_manifest.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=4)
        return p

    def export_inference_manifest(self, data: Dict[str, Any]) -> Path:
        p = self.manifests_dir / "phase9cd_inference_manifest.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=4)
        return p

    def export_candidate_comparison(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        df.to_csv(self.benchmarks_dir / "phase9cd_candidate_comparison.csv", index=False)
        return df
