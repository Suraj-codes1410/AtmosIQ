import shutil
import hashlib
import json
import platform
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import xgboost
import optuna
import shap

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("FreezePackagerPhase4J")


class FreezePackagerPhase4J:
    """
    Production Freeze and Release Packaging Engine for Phase 4J.
    Packages the promoted v3 Random Forest model into ml/models/production/v3/ and ml/releases/v1/.
    """

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir
        self.prod_dir = self.root_dir / "ml" / "models" / "production" / "v3"
        self.release_dir = self.root_dir / "ml" / "releases" / "v1"
        self.src_model_path = self.root_dir / "ml" / "models" / "attribution" / "v3" / "model.joblib"
        self.src_dataset_path = self.root_dir / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"

    @staticmethod
    def calculate_sha256(filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def freeze_production_model(self, features_35: list, v3_model_hash: str, v3_dataset_hash: str) -> dict:
        logger.info("Freezing Production Model Package to ml/models/production/v3/...")
        self.prod_dir.mkdir(parents=True, exist_ok=True)
        self.release_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy model artifact
        dest_model = self.prod_dir / "model.joblib"
        shutil.copy2(self.src_model_path, dest_model)
        frozen_model_hash = self.calculate_sha256(dest_model)
        assert frozen_model_hash == v3_model_hash, "Frozen model hash mismatch during copy!"

        # 2. Feature Registry
        feat_reg_df = pd.DataFrame({
            "feature_order": range(1, len(features_35) + 1),
            "feature_name": features_35,
            "status": "APPROVED_PREDICTION_SAFE",
            "production_input": True
        })
        feat_reg_path = self.prod_dir / "feature_registry.csv"
        feat_reg_df.to_csv(feat_reg_path, index=False)
        feat_reg_hash = self.calculate_sha256(feat_reg_path)

        # 3. Model Manifest
        model_manifest = {
            "model_name": "MODEL_V3_PRODUCTION",
            "model_version": "v3.0.0-frozen",
            "model_type": "random_forest_regressor",
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
            "dataset_version": "Dataset_v3",
            "dataset_sha256": v3_dataset_hash,
            "model_sha256": frozen_model_hash,
            "feature_registry_sha256": feat_reg_hash,
            "feature_count": len(features_35),
            "features": features_35,
            "hyperparameters": {
                "n_estimators": 400,
                "max_depth": 9,
                "min_samples_split": 4,
                "min_samples_leaf": 5,
                "max_features": 0.7,
                "random_state": 42
            },
            "training_period": "2020-01-01 to 2023-12-31",
            "validation_period": "2022-01-01 to 2023-12-31",
            "test_period": "2024-01-01 to 2024-12-31",
            "performance_metrics": {
                "walk_forward_mean_mae": 17.0158,
                "walk_forward_mean_r2": 0.9497,
                "locked_test_2024_mae": 16.8912,
                "delta_mae_vs_v2_control": -8.6428,
                "delta_r2_vs_v2_control": 0.0770,
                "wilcoxon_p_value": 3.5567e-33,
                "bootstrap_95_ci_delta_mae": [-9.7750, -7.2943]
            },
            "scientific_safeguards": {
                "causal_claim_disclaimer": "PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION",
                "persistence_interpretation": "pm25_persistence is a predictive historical-state variable, not an independent physical emission source."
            },
            "freeze_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "FROZEN_PRODUCTION"
        }
        with open(self.prod_dir / "model_manifest.json", "w") as f:
            json.dump(model_manifest, f, indent=4)

        # 4. Dataset Manifest
        dataset_manifest = {
            "dataset_name": "Dataset_v3",
            "dataset_version": "v3.0.0",
            "sha256": v3_dataset_hash,
            "row_count": 1827,
            "total_columns": 275,
            "production_model_input_features": 35,
            "date_range": {
                "start_date": "2020-01-01",
                "end_date": "2024-12-31"
            },
            "publication_status": "PRIVATE_UNPUBLISHED_RELEASE_CANDIDATE",
            "intended_use": "Research forecasting, environmental attribution, and counterfactual sensitivity evaluation."
        }
        with open(self.prod_dir / "dataset_manifest.json", "w") as f:
            json.dump(dataset_manifest, f, indent=4)

        # 5. Environment Metadata
        env_meta = {
            "python_version": platform.python_version(),
            "system_os": platform.system(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "optuna_version": optuna.__version__,
            "shap_version": shap.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__
        }
        with open(self.prod_dir / "environment.json", "w") as f:
            json.dump(env_meta, f, indent=4)

        # 6. Checksums for Production Package
        prod_checksums = []
        for p in sorted(self.prod_dir.glob("*.*")):
            if p.is_file():
                h = self.calculate_sha256(p)
                prod_checksums.append(f"{h}  {p.name}")
        with open(self.prod_dir / "checksums.txt", "w") as f:
            f.write("\n".join(prod_checksums) + "\n")

        # 7. README & RELEASE_NOTES
        readme_content = f"""# AtmosIQ Frozen Production Model (v3)

## Overview
This directory contains the authoritative, frozen production model for the AtmosIQ Delhi NCR PM2.5 forecasting platform.

- **Model Version**: `MODEL_V3_PRODUCTION` (`v3.0.0-frozen`)
- **Model Type**: Random Forest Regressor (`scikit-learn` {sklearn.__version__})
- **Feature Set**: `Candidate_C_V3_Compact` (35 prediction-safe features)
- **Model SHA-256**: `{frozen_model_hash}`
- **Dataset SHA-256**: `{v3_dataset_hash}`
- **Performance**: 3-Fold Walk-Forward MAE = 17.0158 µg/m³, R² = 0.9497

## Scientific Disclaimer
> **PREDICTIVE IMPORTANCE ≠ SHAP ATTRIBUTION ≠ COUNTERFACTUAL MODEL RESPONSE ≠ CAUSAL EFFECT ≠ ACTUAL EMISSION CONTRIBUTION**
"""
        with open(self.prod_dir / "README.md", "w") as f:
            f.write(readme_content)

        release_notes = f"""# AtmosIQ Production Release Notes (v3.0.0)

## Promotion Summary
- **Promoted Candidate**: Random Forest Regressor trained on Dataset v3 with 35 prediction-safe features.
- **Improvement over Phase 3G Control**: ΔMAE = -8.6428 µg/m³ (p = 3.5567e-33, 95% Bootstrap CI: [-9.7750, -7.2943] µg/m³).
- **Extreme Event Improvement**: 17.44 µg/m³ error reduction on PM2.5 >= 150 µg/m³ events.
- **Attribution Revalidation**: TreeSHAP reconstruction error <= 1e-12 µg/m³, 94.73% active-driver counterfactual consistency.
"""
        with open(self.prod_dir / "RELEASE_NOTES.md", "w") as f:
            f.write(release_notes)

        # 8. Master Release Manifest in ml/releases/v1/
        release_manifest_data = {
            "release_name": "AtmosIQ-v3.0.0-Production-Release-Candidate",
            "release_version": "v3.0.0",
            "production_model": {
                "name": "MODEL_V3_PRODUCTION",
                "sha256": frozen_model_hash,
                "library": "scikit-learn",
                "feature_count": 35,
                "performance_mae": 17.0158,
                "performance_r2": 0.9497
            },
            "datasets": {
                "v1_historical": self.calculate_sha256(self.root_dir / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"),
                "v2_production_control": self.calculate_sha256(self.root_dir / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"),
                "v3_authoritative_production": v3_dataset_hash
            },
            "control_models": {
                "MODEL_V2_PRODUCTION_CONTROL": self.calculate_sha256(self.root_dir / "ml" / "models" / "attribution" / "v1" / "model.joblib")
            },
            "authoritative_baselines": {
                "full_population_count": 1827,
                "full_population_mean_observed_pm25": 142.8854,
                "full_population_mean_predicted_pm25": 143.0217,
                "combined_biomass_wind_delta_pm25": -5.0990,
                "combined_all_favorable_delta_pm25": -5.4520,
                "active_driver_directional_consistency_pct": 94.73
            },
            "publication_status": {
                "dataset_v3": "PRIVATE_UNPUBLISHED_RELEASE_CANDIDATE",
                "model_v3": "FROZEN_LOCAL_PRODUCTION",
                "public_release_blockers": [
                    "Complete final user validation review before public Kaggle/GitHub release"
                ],
                "public_release_ready": False
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        with open(self.release_dir / "release_manifest.json", "w") as f:
            json.dump(release_manifest_data, f, indent=4)

        logger.info(f"Production Package Frozen cleanly at {self.prod_dir} and {self.release_dir}.")
        return {
            "prod_dir": self.prod_dir,
            "release_dir": self.release_dir,
            "frozen_model_hash": frozen_model_hash
        }
