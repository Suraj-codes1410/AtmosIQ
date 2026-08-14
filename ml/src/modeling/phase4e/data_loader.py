import sys
import json
import joblib
import hashlib
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("DataLoaderPhase4E")

EXPECTED_V1_HASH = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
EXPECTED_V2_HASH = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
EXPECTED_MODEL_HASH = "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class DataLoaderPhase4E:
    """
    Centralized data loader and cache layer for AtmosIQ Phase 4E.
    Performs mandatory artifact integrity verification on startup.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataLoaderPhase4E, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        base_dir: Path = ROOT_DIR,
        verify_integrity: bool = True
    ):
        if self._initialized:
            return

        self.base_dir = Path(base_dir)
        self.modeling_v1_dir = self.base_dir / "ml" / "data" / "modeling" / "v1"
        self.modeling_v2_dir = self.base_dir / "ml" / "data" / "modeling" / "v2"
        self.pkg_dir = self.base_dir / "ml" / "models" / "attribution" / "v1"
        self.exp_4b_dir = self.base_dir / "ml" / "experiments" / "phase4b"
        self.exp_4c_dir = self.base_dir / "ml" / "experiments" / "phase4c"
        self.exp_4d_dir = self.base_dir / "ml" / "experiments" / "phase4d"

        if verify_integrity:
            self._verify_artifact_integrity()

        self._load_and_cache_all()
        self._initialized = True

    def _verify_artifact_integrity(self):
        """Verifies immutable SHA-256 hashes of Dataset v1, Dataset v2, and Frozen Model."""
        logger.info("Performing mandatory Phase 4E Artifact Integrity Verification...")

        v1_path = self.modeling_v1_dir / "feature_dataset_frozen.csv"
        v2_path = self.modeling_v2_dir / "feature_dataset_frozen.csv"
        model_path = self.pkg_dir / "model.joblib"

        assert v1_path.exists(), f"Dataset v1 missing at {v1_path}!"
        assert v2_path.exists(), f"Dataset v2 missing at {v2_path}!"
        assert model_path.exists(), f"Production model missing at {model_path}!"

        v1_hash = calculate_sha256(v1_path)
        v2_hash = calculate_sha256(v2_path)
        model_hash = calculate_sha256(model_path)

        if v1_hash != EXPECTED_V1_HASH:
            raise RuntimeError(f"DATASET INTEGRITY FAILURE: Dataset v1 hash mismatch! Expected {EXPECTED_V1_HASH}, got {v1_hash}")

        if v2_hash != EXPECTED_V2_HASH:
            raise RuntimeError(f"DATASET INTEGRITY FAILURE: Dataset v2 hash mismatch! Expected {EXPECTED_V2_HASH}, got {v2_hash}")

        if model_hash != EXPECTED_MODEL_HASH:
            raise RuntimeError(f"MODEL INTEGRITY FAILURE: Model hash mismatch! Expected {EXPECTED_MODEL_HASH}, got {model_hash}")

        logger.info("Artifact Integrity Verification: 100% PASS.")

    def _load_and_cache_all(self):
        """Loads and caches all required models, datasets, registries, and experiment outputs."""
        logger.info("Loading Phase 3G-4D upstream artifacts into memory cache...")

        # 1. Model & Registries
        self.model = joblib.load(self.pkg_dir / "model.joblib")
        self.feat_reg_df = pd.read_csv(self.pkg_dir / "feature_registry.csv")
        self.attr_groups_df = pd.read_csv(self.pkg_dir / "attribution_groups.csv")

        # 2. Dataset v2
        self.df_v2 = pd.read_csv(self.modeling_v2_dir / "feature_dataset_frozen.csv")
        self.feature_names = self.feat_reg_df["feature_name"].tolist()
        self.X_v2 = self.df_v2[self.feature_names]

        # 3. Phase 4B SHAP outputs
        self.shap_summary_df = pd.read_csv(self.exp_4b_dir / "summaries" / "global_feature_importance.csv")
        self.group_shap_df = pd.read_csv(self.exp_4b_dir / "group_attributions" / "group_attributions_daily.csv") if (self.exp_4b_dir / "group_attributions" / "group_attributions_daily.csv").exists() else pd.DataFrame()

        # 4. Phase 4C outputs
        self.event_catalog_df = pd.read_csv(self.exp_4c_dir / "event_catalog.csv")
        self.conflicts_df = pd.read_csv(self.exp_4c_dir / "attribution_conflicts.csv")
        self.conf_4c_df = pd.read_csv(self.exp_4c_dir / "confidence_scores.csv")
        self.stat_tests_df = pd.read_csv(self.exp_4c_dir / "statistical_tests.csv")

        # 5. Phase 4D outputs
        self.cf_results_df = pd.read_csv(self.exp_4d_dir / "counterfactual_results.csv")
        self.cf_summary_df = pd.read_csv(self.exp_4d_dir / "group_counterfactual_summary.csv")
        self.interaction_df = pd.read_csv(self.exp_4d_dir / "interaction_analysis.csv")
        self.event_cf_df = pd.read_csv(self.exp_4d_dir / "event_counterfactuals.csv")
        self.plausibility_df = pd.read_csv(self.exp_4d_dir / "plausibility_checks.csv")
        self.ood_df = pd.read_csv(self.exp_4d_dir / "ood_analysis.csv")
        self.conf_4d_df = pd.read_csv(self.exp_4d_dir / "confidence_scores.csv")
        self.shap_cf_consistency_df = pd.read_csv(self.exp_4d_dir / "shap_counterfactual_consistency.csv")

        with open(self.exp_4d_dir / "scenario_registry.json", "r") as f:
            self.scenario_registry = json.load(f)

        logger.info("All Phase 3G-4D upstream artifacts successfully cached.")


if __name__ == "__main__":
    loader = DataLoaderPhase4E()
    print("DataLoader initialized successfully.")
