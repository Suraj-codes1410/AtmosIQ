import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("StabilityAnalysisPhase4H")


class StabilityAnalysisPhase4H:
    """
    Year-to-Year Performance Stability Analysis Engine for Phase 4H.
    Evaluates out-of-sample performance variance and drift across 2022, 2023, and 2024.
    """

    def run_stability_eval(self, fold_results_list: list) -> pd.DataFrame:
        """
        fold_results_list: list of fold dicts containing keys:
        ['model_name', 'feature_set', 'test_year', 'test_mae', 'test_rmse', 'test_r2']
        """
        logger.info("Executing Year-to-Year Stability Analysis...")

        df_folds = pd.DataFrame(fold_results_list)
        records = []

        grouped = df_folds.groupby(["model_name", "feature_set"])
        for (m_name, f_set), group in grouped:
            maes = group['test_mae'].values
            rmses = group['test_rmse'].values
            r2s = group['test_r2'].values
            years = group['test_year'].values

            records.append({
                "model_name": m_name,
                "feature_set": f_set,
                "mae_2022": round(float(group[group['test_year'] == 2022]['test_mae'].values[0]), 4) if 2022 in years else None,
                "mae_2023": round(float(group[group['test_year'] == 2023]['test_mae'].values[0]), 4) if 2023 in years else None,
                "mae_2024": round(float(group[group['test_year'] == 2024]['test_mae'].values[0]), 4) if 2024 in years else None,
                "mean_mae": round(float(np.mean(maes)), 4),
                "std_mae": round(float(np.std(maes)), 4),
                "min_mae": round(float(np.min(maes)), 4),
                "max_mae": round(float(np.max(maes)), 4),
                "mae_drift_max_minus_min": round(float(np.max(maes) - np.min(maes)), 4),
                "mean_r2": round(float(np.mean(r2s)), 4),
                "std_r2": round(float(np.std(r2s)), 4)
            })

        return pd.DataFrame(records)
