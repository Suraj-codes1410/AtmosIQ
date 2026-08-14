import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4g.feature_sets import get_feature_sets
from ml.src.modeling.phase4g.walk_forward import WalkForwardPhase4G

logger = setup_logger("ExperimentsPhase4G")


class ExperimentsPhase4G:
    """
    Experimental Grid Engine for Phase 4G.
    Executes walk-forward evaluations across Models x Feature Sets x Folds.
    """

    def run_grid_experiments(self, v3_df: pd.DataFrame, output_dir: Path):
        logger.info("Executing Phase 4G Model Grid & Walk-Forward Benchmark Experiments...")

        feature_sets = get_feature_sets(v3_df)
        wf_engine = WalkForwardPhase4G(v3_df)

        models = ["Persistence", "Ridge", "ElasticNet", "RandomForest", "XGBoost"]

        detailed_results = []
        summary_results = []
        raw_predictions_store = {}

        for fs_name, f_list in feature_sets.items():
            logger.info(f"Evaluating Feature Set: {fs_name} ({len(f_list)} features)...")

            for m_name in models:
                fold_metrics = []

                for fold_info in wf_engine.folds:
                    res = wf_engine.evaluate_model_on_fold(m_name, fs_name, f_list, fold_info)

                    # Store detailed fold result
                    detailed_results.append({
                        "fold": res["fold"],
                        "train_years": res["train_years"],
                        "test_year": res["test_year"],
                        "model_name": res["model_name"],
                        "feature_set": res["feature_set"],
                        "num_features": res["num_features"],
                        "train_r2": round(res["train_r2"], 4),
                        "test_r2": round(res["test_r2"], 4),
                        "test_mae": round(res["test_mae"], 4),
                        "test_rmse": round(res["test_rmse"], 4),
                        "test_medae": round(res["test_medae"], 4),
                        "generalization_gap": round(res["generalization_gap"], 4)
                    })

                    fold_metrics.append(res)
                    key = f"{m_name}__{fs_name}__Fold{res['fold']}"
                    raw_predictions_store[key] = (res["y_test"], res["y_pred_test"])

                # Average across folds for summary
                mean_test_r2 = np.mean([f["test_r2"] for f in fold_metrics])
                mean_test_mae = np.mean([f["test_mae"] for f in fold_metrics])
                mean_test_rmse = np.mean([f["test_rmse"] for f in fold_metrics])
                mean_gen_gap = np.mean([f["generalization_gap"] for f in fold_metrics])

                summary_results.append({
                    "model_name": m_name,
                    "feature_set": fs_name,
                    "num_features": len(f_list),
                    "mean_test_r2": round(float(mean_test_r2), 4),
                    "mean_test_mae": round(float(mean_test_mae), 4),
                    "mean_test_rmse": round(float(mean_test_rmse), 4),
                    "mean_generalization_gap": round(float(mean_gen_gap), 4)
                })

        df_detailed = pd.DataFrame(detailed_results)
        df_summary = pd.DataFrame(summary_results)

        csv_wf = output_dir / "walk_forward_results_v3.csv"
        csv_metrics = output_dir / "model_metrics_v3.csv"

        df_detailed.to_csv(csv_wf, index=False)
        df_summary.to_csv(csv_metrics, index=False)

        logger.info(f"Walk-forward detailed results saved to {csv_wf}.")
        logger.info(f"Model metrics summary saved to {csv_metrics}.")

        return df_summary, df_detailed, raw_predictions_store
