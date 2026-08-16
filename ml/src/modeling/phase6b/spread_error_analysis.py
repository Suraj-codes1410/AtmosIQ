import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("SpreadErrorAnalysisPhase6B")


class SpreadErrorAnalysisEnginePhase6B:
    """
    Spread vs Actual Error Correlation and Uncertainty Discrimination Engine for Phase 6B.
    Evaluates whether ensemble spread behaves as a meaningful predictor of prediction error.
    """

    def __init__(self, df_ensemble_preds: pd.DataFrame, ensemble_name: str = "bootstrap"):
        self.df = df_ensemble_preds[df_ensemble_preds['ensemble_type'] == ensemble_name].copy()
        self.ensemble_name = ensemble_name

    def run_spread_error_analysis(self, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        logger.info(f"Executing Spread vs Error Correlation Analysis for {self.ensemble_name}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        spread = self.df['ensemble_std'].values
        abs_err = self.df['absolute_error'].values
        sq_err = self.df['squared_error'].values

        # 1. Statistical Correlations
        pearson_r, pearson_p = stats.pearsonr(spread, abs_err)
        spearman_rho, spearman_p = stats.spearmanr(spread, abs_err)
        pearson_sq_r, pearson_sq_p = stats.pearsonr(spread, sq_err)
        spearman_sq_rho, spearman_sq_p = stats.spearmanr(spread, sq_err)

        correlation_stats = {
            "ensemble_type": self.ensemble_name,
            "pearson_r_abs_error": float(pearson_r),
            "pearson_p_abs_error": float(pearson_p),
            "spearman_rho_abs_error": float(spearman_rho),
            "spearman_p_abs_error": float(spearman_p),
            "pearson_r_sq_error": float(pearson_sq_r),
            "spearman_rho_sq_error": float(spearman_sq_rho)
        }

        # 2. Quintile Analysis (Q1 - Q5)
        self.df['spread_quintile'] = pd.qcut(self.df['ensemble_std'], q=5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"])
        
        quintile_records = []
        for q_name, group in self.df.groupby('spread_quintile', observed=False):
            n = len(group)
            m_spread = float(group['ensemble_std'].mean())
            med_spread = float(group['ensemble_std'].median())
            mae_val = float(group['absolute_error'].mean())
            rmse_val = float(np.sqrt((group['residual'] ** 2).mean()))
            medae_val = float(group['absolute_error'].median())

            # 90% coverage within quintile
            q05 = group['q05'].values
            q95 = group['q95'].values
            obs = group['observed_pm25'].values
            cov_90 = float(np.mean((obs >= q05) & (obs <= q95)))

            quintile_records.append({
                "ensemble_type": self.ensemble_name,
                "spread_quintile": q_name,
                "count": n,
                "mean_spread_ugm3": m_spread,
                "median_spread_ugm3": med_spread,
                "mae_ugm3": mae_val,
                "rmse_ugm3": rmse_val,
                "medae_ugm3": medae_val,
                "empirical_coverage_90pct": cov_90
            })

        df_quintiles = pd.DataFrame(quintile_records)
        df_quintiles.to_csv(output_dir / "spread_error_analysis.csv", index=False)

        # 3. Uncertainty Discrimination Test
        # High error threshold: >= 90th percentile of absolute error
        err_thresh_90 = float(np.percentile(abs_err, 90.0))
        err_thresh_80 = float(np.percentile(abs_err, 80.0))

        y_true_high_90 = (abs_err >= err_thresh_90).astype(int)
        y_true_high_80 = (abs_err >= err_thresh_80).astype(int)

        roc_auc_90 = float(roc_auc_score(y_true_high_90, spread))
        roc_auc_80 = float(roc_auc_score(y_true_high_80, spread))

        # Precision & Recall at Top K% spread
        spread_thresh_top10 = float(np.percentile(spread, 90.0))
        spread_thresh_top20 = float(np.percentile(spread, 80.0))

        top10_mask = (spread >= spread_thresh_top10)
        top20_mask = (spread >= spread_thresh_top20)

        prec_top10 = float(np.mean(y_true_high_90[top10_mask])) if top10_mask.sum() > 0 else 0.0
        recall_top10 = float(np.sum(y_true_high_90[top10_mask]) / np.sum(y_true_high_90)) if np.sum(y_true_high_90) > 0 else 0.0

        prec_top20 = float(np.mean(y_true_high_90[top20_mask])) if top20_mask.sum() > 0 else 0.0
        recall_top20 = float(np.sum(y_true_high_90[top20_mask]) / np.sum(y_true_high_90)) if np.sum(y_true_high_90) > 0 else 0.0

        precision_arr, recall_arr, _ = precision_recall_curve(y_true_high_90, spread)
        pr_auc_90 = float(auc(recall_arr, precision_arr))

        discrimination_records = [
            {
                "ensemble_type": self.ensemble_name,
                "error_threshold_definition": "Top 10% Absolute Error (>= 90th percentile)",
                "error_threshold_ugm3": err_thresh_90,
                "roc_auc": roc_auc_90,
                "pr_auc": pr_auc_90,
                "precision_at_top_10pct_spread": prec_top10,
                "recall_at_top_10pct_spread": recall_top10,
                "precision_at_top_20pct_spread": prec_top20,
                "recall_at_top_20pct_spread": recall_top20,
                "baseline_random_precision": float(np.mean(y_true_high_90))
            },
            {
                "ensemble_type": self.ensemble_name,
                "error_threshold_definition": "Top 20% Absolute Error (>= 80th percentile)",
                "error_threshold_ugm3": err_thresh_80,
                "roc_auc": roc_auc_80,
                "pr_auc": float(auc(*precision_recall_curve(y_true_high_80, spread)[1::-1])),
                "precision_at_top_10pct_spread": float(np.mean(y_true_high_80[top10_mask])),
                "recall_at_top_10pct_spread": float(np.sum(y_true_high_80[top10_mask]) / np.sum(y_true_high_80)),
                "precision_at_top_20pct_spread": float(np.mean(y_true_high_80[top20_mask])),
                "recall_at_top_20pct_spread": float(np.sum(y_true_high_80[top20_mask]) / np.sum(y_true_high_80)),
                "baseline_random_precision": float(np.mean(y_true_high_80))
            }
        ]

        df_disc = pd.DataFrame(discrimination_records)
        df_disc.to_csv(output_dir / "uncertainty_discrimination.csv", index=False)

        logger.info(f"Spread vs Error Complete. Spearman Rho = {spearman_rho:.4f} (p={spearman_p:.2e}), ROC-AUC={roc_auc_90:.4f}")
        return df_quintiles, df_disc, correlation_stats
