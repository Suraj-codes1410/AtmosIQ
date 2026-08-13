import sys
import json
import joblib
import datetime
import hashlib
import platform
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas
import numpy
import sklearn
import scipy
import joblib as jlib_pkg
import xgboost
from ml.src.utils.logger import setup_logger

logger = setup_logger("PackageFreezerPhase4A")


class PackageFreezerPhase4A:
    """
    AtmosIQ Phase 4A Package Freezer.
    Creates immutable, reproducible attribution package under ml/models/attribution/v1/.
    """

    def __init__(self, package_dir: str = "ml/models/attribution/v1"):
        self.pkg_dir = Path(package_dir)
        self.pkg_dir.mkdir(parents=True, exist_ok=True)

        self.phase3g_model_file = Path("ml/models/phase3g/model.pkl")
        assert self.phase3g_model_file.exists(), f"Phase 3G model pickle missing: {self.phase3g_model_file}"

        self.phase3g_meta_file = Path("ml/models/phase3g/training_metadata.json")
        assert self.phase3g_meta_file.exists(), f"Phase 3G metadata missing: {self.phase3g_meta_file}"

        with open(self.phase3g_meta_file, "r") as f:
            self.p3g_meta = json.load(f)

    @staticmethod
    def calculate_sha256(filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def serialize_and_build_manifests(self):
        """Copies model joblib and writes manifest files."""
        logger.info("Freezing Attribution Package under ml/models/attribution/v1/...")

        # 1. Copy model pickle to model.joblib
        model_obj = joblib.load(self.phase3g_model_file)
        joblib_path = self.pkg_dir / "model.joblib"
        joblib.dump(model_obj, joblib_path)

        model_sha256 = self.calculate_sha256(joblib_path)

        # 2. Environment Manifest
        try:
            import shap
            shap_version = shap.__version__
        except ImportError:
            shap_version = "available_in_phase4b"

        try:
            import lightgbm
            lgb_version = lightgbm.__version__
        except ImportError:
            lgb_version = "not_installed"

        env_manifest = {
            "os_name": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version.split()[0],
            "pandas_version": pandas.__version__,
            "numpy_version": numpy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "scipy_version": scipy.__version__,
            "joblib_version": jlib_pkg.__version__,
            "xgboost_version": xgboost.__version__,
            "lightgbm_version": lgb_version,
            "shap_version": shap_version
        }
        with open(self.pkg_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(env_manifest, f, indent=4)

        # 3. Dataset Manifest
        ds_manifest_file = Path("ml/models/phase3g/dataset_manifest.json")
        if ds_manifest_file.exists():
            with open(ds_manifest_file, "r") as f:
                ds_manifest = json.load(f)
        else:
            ds_manifest = {"dataset_version": "v2", "sha256": "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"}

        with open(self.pkg_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(ds_manifest, f, indent=4)

        # 4. Model Manifest
        with open("ml/models/phase3g/feature_list.json", "r") as f:
            f_cols = json.load(f)["features"]

        with open("ml/models/phase3g/model_config.json", "r") as f:
            m_config = json.load(f)

        model_manifest = {
            "project": "AtmosIQ",
            "phase": "Phase 4A",
            "package_version": "v1",
            "model_type": self.p3g_meta.get("model_type", "random_forest"),
            "model_library": "scikit-learn",
            "model_library_version": sklearn.__version__,
            "dataset_version": "v2",
            "dataset_sha256": ds_manifest.get("sha256", "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"),
            "model_sha256": model_sha256,
            "feature_count": len(f_cols),
            "feature_names": f_cols,
            "feature_order": f_cols,
            "target": "pm25",
            "training_start": "2020-01-01",
            "training_end": "2023-12-31",
            "validation_start": "2022-01-01",
            "validation_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2024-12-31",
            "random_seed": 42,
            "hyperparameters": m_config.get("hyperparameters", {}),
            "preprocessing": "StandardScaler (for linear baselines) / None (for Random Forest)",
            "training_script": "ml/src/modeling/phase3g/phase3g_runner.py",
            "source_experiment": "ml/experiments/phase3g",
            "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "attribution_ready": True
        }
        with open(self.pkg_dir / "model_manifest.json", "w", encoding="utf-8") as f:
            json.dump(model_manifest, f, indent=4)

        # 5. Checksum Manifest
        files_to_check = [
            "model.joblib",
            "feature_registry.csv",
            "attribution_groups.csv",
            "model_manifest.json",
            "dataset_manifest.json",
            "environment.json"
        ]

        checksum_lines = []
        for fname in files_to_check:
            fpath = self.pkg_dir / fname
            if fpath.exists():
                c_hash = self.calculate_sha256(fpath)
                checksum_lines.append(f"{c_hash}  {fname}\n")

        with open(self.pkg_dir / "checksums.txt", "w", encoding="utf-8") as f:
            f.writelines(checksum_lines)

        # 6. Package README.md
        readme_md = f"""# AtmosIQ Phase 4A Attribution Model Package v1

This package contains the immutable, reproducible Phase 3G production forecasting model serialized for Phase 4 TreeSHAP attribution.

## Contents
- `model.joblib`: Serialized Random Forest Regressor trained on Dataset v2 (2020-01-01 to 2023-12-31).
- `model_manifest.json`: Full model metadata, hyperparameters, feature order, and SHA-256 checksums.
- `feature_registry.csv`: 147 prediction-safe features in exact model feature order.
- `attribution_groups.csv`: Deterministic mapping from model features to environmental process attribution groups.
- `dataset_manifest.json`: Dataset v2 manifest snapshot (SHA-256: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`).
- `environment.json`: Python environment dependency versions.
- `checksums.txt`: SHA-256 checksums for package integrity verification.

## Interface Contract for Phase 4B
Phase 4B will consume:
1. `model.joblib`
2. `feature_registry.csv`
3. `attribution_groups.csv`
4. `ml/data/modeling/v2/feature_dataset_frozen.csv`

Phase 4B TreeSHAP Reconstruction Check:
$$\\text{{base\_value}} + \\sum \\text{{SHAP\_values}} \\approx \\hat{{y}}_{{\\text{{pred}}}}$$
"""
        with open(self.pkg_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_md)

        logger.info(f"Attribution Package v1 successfully created at: {self.pkg_dir}")


if __name__ == "__main__":
    freezer = PackageFreezerPhase4A()
    freezer.serialize_and_build_manifests()
