import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ModelComparisonPhase4H")


class ModelComparisonEnginePhase4H:
    """
    Master Model Comparison & Consolidator Engine for Phase 4H.
    Generates structured master tables comparing Control v2 and Candidate v3 models.
    """

    def consolidate_results(self, all_fold_results: list, control_summary: dict) -> tuple:
        logger.info("Consolidating Master Model Comparison Tables...")

        df_folds = pd.DataFrame(all_fold_results)

        # Summary per model and feature set
        summary_records = []
        grouped = df_folds.groupby(["model_name", "dataset_version", "feature_set"])

        for (m_name, ds_ver, f_set), group in grouped:
            mean_mae = float(np.mean(group['test_mae']))
            mean_rmse = float(np.mean(group['test_rmse']))
            mean_r2 = float(np.mean(group['test_r2']))
            mean_medae = float(np.mean(group['test_medae']))
            mean_gen_gap = float(np.mean(group['generalization_gap']))

            # Deltas vs Control
            delta_mae = mean_mae - control_summary['mean_mae']
            delta_rmse = mean_rmse - control_summary['mean_rmse']
            delta_r2 = mean_r2 - control_summary['mean_r2']
            delta_medae = mean_medae - control_summary['mean_medae']

            summary_records.append({
                "model_name": m_name,
                "dataset_version": ds_ver,
                "feature_set": f_set,
                "num_features": group['num_features'].iloc[0],
                "mean_mae": round(mean_mae, 4),
                "mean_rmse": round(mean_rmse, 4),
                "mean_r2": round(mean_r2, 4),
                "mean_median_ae": round(mean_medae, 4),
                "mean_generalization_gap": round(mean_gen_gap, 4),
                "delta_mae_vs_v2": round(delta_mae, 4),
                "delta_rmse_vs_v2": round(delta_rmse, 4),
                "delta_r2_vs_v2": round(delta_r2, 4),
                "delta_median_ae_vs_v2": round(delta_medae, 4)
            })

        df_summary = pd.DataFrame(summary_records).sort_values("mean_mae").reset_index(drop=True)
        return df_summary, df_folds
