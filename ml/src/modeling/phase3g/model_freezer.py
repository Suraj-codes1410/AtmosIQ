import sys
import json
import joblib
import datetime
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sklearn
import xgboost as xgb
import optuna
from ml.src.utils.logger import setup_logger

logger = setup_logger("ModelFreezerPhase3G")


class ModelFreezerPhase3G:
    """
    AtmosIQ Phase 3G Production Model Freezer.
    Freezes the selected final production model configuration, fitted weights, feature lists, and metadata under ml/models/phase3g/.
    """

    def __init__(self, models_dir: str = "ml/models/phase3g"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def freeze_production_model(self, final_results: dict):
        """Freezes selected final model and metadata."""
        logger.info("Freezing selected final production forecasting model under ml/models/phase3g/...")

        best_cand = final_results["best_candidate"]
        best_info = final_results["best_info"]
        final_model = final_results["final_model"]
        test_m = final_results["test_metrics"]
        pers_m = final_results["persistence_test_metrics"]
        pct_impr = final_results["pct_improvement"]
        f_cols = final_results["feature_cols"]

        # 1. Save fitted model pickle
        model_pkl_path = self.models_dir / "model.pkl"
        joblib.dump(final_model, model_pkl_path)

        # 2. Save feature_list.json
        feature_list_path = self.models_dir / "feature_list.json"
        with open(feature_list_path, "w", encoding="utf-8") as f:
            json.dump({"feature_set": best_cand["Feature_Set"], "feature_count": len(f_cols), "features": f_cols}, f, indent=4)

        # 3. Save model_config.json
        config_path = self.models_dir / "model_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_type": best_info["model_type"],
                "model_name": best_cand["Model"],
                "feature_set": best_cand["Feature_Set"],
                "hyperparameters": best_info["params"]
            }, f, indent=4)

        # 4. Save metrics.json
        metrics_path = self.models_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "dev_walk_forward_mean_mae": best_cand["Dev_Mean_MAE"],
                "dev_walk_forward_mae_std": best_cand["Dev_MAE_Std"],
                "final_test_2024_mae": test_m["MAE"],
                "final_test_2024_rmse": test_m["RMSE"],
                "final_test_2024_r2": test_m["R2"],
                "final_test_2024_median_ae": test_m["Median_AE"],
                "persistence_test_2024_mae": pers_m["MAE"],
                "improvement_vs_persistence_pct": pct_impr
            }, f, indent=4)

        # 5. Load dataset manifest
        manifest_file = Path("ml/data/modeling/v2/dataset_manifest.json")
        if manifest_file.exists():
            with open(manifest_file, "r") as f:
                manifest_data = json.load(f)
        else:
            manifest_data = {"dataset_version": "v2", "sha256": "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"}

        with open(self.models_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4)

        # 6. Save training_metadata.json
        metadata = {
            "model_id": "phase3g_final_production_model",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_type": best_info["model_type"],
            "model_name": best_cand["Model"],
            "feature_set": best_cand["Feature_Set"],
            "feature_count": len(f_cols),
            "training_date_range": "2020-01-01 to 2023-12-31",
            "training_rows": 1461,
            "test_date_range": "2024-01-01 to 2024-12-31",
            "test_rows": 366,
            "dataset_version": "v2",
            "dataset_sha256": manifest_data.get("sha256", ""),
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "optuna_version": optuna.__version__,
            "random_seed": 42,
            "ready_for_shap": True
        }
        with open(self.models_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Production model successfully frozen under: {self.models_dir}")


if __name__ == "__main__":
    freezer = ModelFreezerPhase3G()
