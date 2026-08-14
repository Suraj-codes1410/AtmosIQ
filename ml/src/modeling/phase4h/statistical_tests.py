import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("StatisticalTestsPhase4H")


class StatisticalTestsPhase4H:
    """
    Statistical Significance & Bootstrap Confidence Interval Engine for Phase 4H.
    Executes paired Wilcoxon signed-rank test and 1,000-resample bootstrap CIs
    on out-of-sample daily error differences between Control v2 and Candidate v3.
    """

    def __init__(self, n_bootstraps: int = 1000, random_seed: int = 42):
        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed

    def run_paired_tests(self, control_df: pd.DataFrame, candidate_df: pd.DataFrame, model_name: str, feature_set: str) -> dict:
        """
        Executes paired error comparisons between control and candidate models.
        control_df and candidate_df must contain columns: 'date', 'y_true', 'y_pred'.
        """
        merged = pd.merge(control_df, candidate_df, on=["date", "y_true"], suffixes=("_ctrl", "_cand"))

        y_true = merged["y_true"].values
        ctrl_preds = merged["y_pred_ctrl"].values
        cand_preds = merged["y_pred_cand"].values

        ctrl_errs = np.abs(ctrl_preds - y_true)
        cand_errs = np.abs(cand_preds - y_true)

        err_diff = cand_errs - ctrl_errs  # Negative = candidate is better

        # Paired Wilcoxon Signed-Rank Test
        try:
            stat, p_value = wilcoxon(cand_errs, ctrl_errs, alternative='less')
        except Exception as e:
            logger.warning(f"Wilcoxon test warning: {e}")
            stat, p_value = 0.0, 1.0

        mean_err_diff = float(np.mean(err_diff))
        median_err_diff = float(np.median(err_diff))

        # Bootstrap Confidence Intervals for Metric Differences (ΔMAE, ΔRMSE, ΔR2)
        np.random.seed(self.random_seed)
        n = len(merged)
        delta_maes, delta_rmses, delta_r2s = [], [], []

        for _ in range(self.n_bootstraps):
            indices = np.random.choice(n, size=n, replace=True)
            y_b = y_true[indices]
            ctrl_b = ctrl_preds[indices]
            cand_b = cand_preds[indices]

            mae_ctrl = mean_absolute_error(y_b, ctrl_b)
            mae_cand = mean_absolute_error(y_b, cand_b)
            delta_maes.append(mae_cand - mae_ctrl)

            rmse_ctrl = root_mean_squared_error(y_b, ctrl_b)
            rmse_cand = root_mean_squared_error(y_b, cand_b)
            delta_rmses.append(rmse_cand - rmse_ctrl)

            r2_ctrl = r2_score(y_b, ctrl_b)
            r2_cand = r2_score(y_b, cand_b)
            delta_r2s.append(r2_cand - r2_ctrl)

        mae_ci_lower = float(np.percentile(delta_maes, 2.5))
        mae_ci_upper = float(np.percentile(delta_maes, 97.5))

        rmse_ci_lower = float(np.percentile(delta_rmses, 2.5))
        rmse_ci_upper = float(np.percentile(delta_rmses, 97.5))

        r2_ci_lower = float(np.percentile(delta_r2s, 2.5))
        r2_ci_upper = float(np.percentile(delta_r2s, 97.5))

        stat_sig = (p_value < 0.05) and (mae_ci_upper < 0.0)

        result = {
            "model_name": model_name,
            "feature_set": feature_set,
            "sample_size_n": n,
            "mean_error_difference": round(mean_err_diff, 4),
            "median_error_difference": round(median_err_diff, 4),
            "wilcoxon_statistic": round(float(stat), 4),
            "p_value": float(p_value),
            "p_value_formatted": f"{p_value:.4e}" if p_value < 1e-4 else f"{p_value:.4f}",
            "delta_mae_mean": round(float(np.mean(delta_maes)), 4),
            "delta_mae_ci_lower": round(mae_ci_lower, 4),
            "delta_mae_ci_upper": round(mae_ci_upper, 4),
            "delta_rmse_mean": round(float(np.mean(delta_rmses)), 4),
            "delta_rmse_ci_lower": round(rmse_ci_lower, 4),
            "delta_rmse_ci_upper": round(rmse_ci_upper, 4),
            "delta_r2_mean": round(float(np.mean(delta_r2s)), 4),
            "delta_r2_ci_lower": round(r2_ci_lower, 4),
            "delta_r2_ci_upper": round(r2_ci_upper, 4),
            "statistically_significant": stat_sig
        }

        logger.info(f"Statistical Test for {model_name} ({feature_set}): mean ΔMAE={mean_err_diff:.4f}, 95% CI=[{mae_ci_lower:.4f}, {mae_ci_upper:.4f}], p={result['p_value_formatted']}, sig={stat_sig}")
        return result
