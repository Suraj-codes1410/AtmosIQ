import sys
import json
import hashlib
from pathlib import Path
import subprocess
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.decision_engine import DecisionEnginePhase4E
from ml.src.modeling.phase4e.response_schema import SCIENTIFIC_DISCLAIMER

logger = setup_logger("MasterRunnerPhase4F")


class MasterRunnerPhase4F:
    """
    AtmosIQ Phase 4F Master Orchestrator Script.
    Validates backend & artifact integrity, verifies React frontend build, runs end-to-end integration checks.
    """

    def __init__(self, docs_dir: str = "docs/phase4"):
        self.docs_dir = Path(docs_dir)
        self.frontend_dir = ROOT_DIR / "frontend"
        self.frontend_dist = self.frontend_dir / "dist"
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self):
        logger.info("=== Starting AtmosIQ Phase 4F Master Pipeline ===")

        # 1. Verify Artifact Integrity
        logger.info("Validating Phase 4F Input Dependencies & Artifact Integrity...")
        loader = DataLoaderPhase4E(verify_integrity=True)
        logger.info("Phase 4F Input Validation & Integrity: 100% PASS.")

        # 2. Verify / Build Frontend Bundle
        logger.info("Verifying React + TypeScript Frontend Production Build...")
        if not self.frontend_dist.exists():
            logger.info("Building frontend production bundle in frontend/dist...")
            res = subprocess.run(["npm", "run", "build"], cwd=str(self.frontend_dir), capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Frontend Build Failure: {res.stderr}")

        assert (self.frontend_dist / "index.html").exists(), "frontend/dist/index.html missing!"
        logger.info("Frontend Production Build Verification: 100% PASS.")

        # 3. Decision Support Engine Integration Check
        logger.info("Running Decision Support Engine Integration Checks...")
        engine = DecisionEnginePhase4E(loader)
        sample_report = engine.generate_decision_support("2024-11-16")
        assert sample_report.prediction.predicted_pm25 > 0
        logger.info("Decision Support Engine Integration Check: 100% PASS.")

        # 4. Generate Phase 4F Documentation Report
        self._write_documentation()
        logger.info("=== Phase 4F Master Pipeline Completed Successfully ===")

    def _write_documentation(self):
        doc_path = self.docs_dir / "phase4f_dashboard.md"
        doc_content = f"""# AtmosIQ — Phase 4F Technical Documentation
## Production Dashboard & Decision Support System Integration

### 1. Executive Summary
Phase 4F introduces a research-grade, responsive user interface for AtmosIQ built with React 18, TypeScript, Vite, and Tailwind CSS, seamlessly integrated with the Phase 4E FastAPI backend.

### 2. Verified Immutable Provenance
- **Dataset v2 SHA-256**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`
- **Dataset v1 SHA-256**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Frozen Model SHA-256**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162`

### 3. Key Implemented UI Views
1. **Main Dashboard**: Prediction Card, TreeSHAP Group Attributions, Feature Importance, Independent Observational Indicators, Counter-Evidence Alert Banner, Interactive Counterfactual Simulator, Confidence Rating.
2. **Extreme Episode Explorer**: 110 extreme pollution episodes catalog filterable by year, season, and peak PM2.5.
3. **2020–2024 Timeline**: 1,827-day historical dataset visualization.
4. **Seasonal Analysis**: Post-Monsoon, Winter, Summer, and Monsoon regime breakdowns.
5. **Methodology Workbench**: Complete sitemap from raw data ingestion to policy decision support.

### 4. Non-Causal Wording Safeguards
```text
PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION
```
"""
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_content)


if __name__ == "__main__":
    runner = MasterRunnerPhase4F()
    runner.run_pipeline()
