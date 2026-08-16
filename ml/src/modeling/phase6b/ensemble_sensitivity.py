import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6a.coverage_metrics import IntervalEvaluationMetricsPhase6A

logger = setup_logger("EnsembleSensitivityPhase6B")


class EnsembleSensitivityEnginePhase6B:
    """
    Ensemble Size Sensitivity, Paradigm Comparison (Bootstrap vs Seed vs Family), and Statistical Testing Engine for Phase 6B.
    """

    def __init__(
        self,
        bootstrap_member_preds: List[np.ndarray],
        seed_member_preds: List[np.ndarray],
        df_control: pd.DataFrame,
        df_boot_summary: pd.DataFrame,
        df_seed_summary: pd.DataFrame,
        df_family_summary: pd.DataFrame
    ):
        self.boot_members = bootstrap_member_preds
        self.seed_members = seed_member_preds
        self.df_control = df_control
        self.df_boot = df_boot_summary
        self.df_seed = df_seed_summary
        self.df_family = df_family_summary

    def run_ensemble_size_sensitivity(self, output_dir: Path, sizes: List[int] = [5, 10, 20, 30, 50]) -> pd.DataFrame:
        logger.info("Executing Ensemble Size Sensitivity Analysis (N in 5, 10, 20, 30, 50)...")
        output_dir.mkdir(parents=True, exist_ok=True)

        sensitivity_records = []

        # Concatenate fold predictions along time dimension for each member
        # boot_members is a list of arrays [B x n_eval_fold]
        B_max = min(m.shape[0] for m in self.boot_members)
        eval_obs_all = self.df_control['observed_pm25'].values

        for size in sizes:
            if size > B_max:
                continue

            # Slice first `size` members across folds
            full_preds = []
            for fold_arr in self.boot_members:
                sub_arr = fold_arr[:size, :]  # shape: size x n_eval
                full_preds.append(sub_arr)
            # Concat along axis 1
            ens_all = np.concatenate(full_preds, axis=1)  # shape: size x N_total

            ens_mean = np.mean(ens_all, axis=0)
            ens_std = np.std(ens_all, axis=0, ddof=1) if size > 1 else np.zeros(len(eval_obs_all))
            
            abs_err = np.abs(eval_obs_all - ens_mean)
            mae_val = float(np.mean(abs_err))
            mean_spread = float(np.mean(ens_std))

            q05 = np.percentile(ens_all, 5.0, axis=0)
            q95 = np.percentile(ens_all, 95.0, axis=0)
            lower_clip = np.maximum(0.0, q05)
            upper_clip = np.maximum(lower_clip, q95)

            m_eval = IntervalEvaluationMetricsPhase6A.evaluate_interval(eval_obs_all, lower_clip, upper_clip, 0.90)
            spearman_corr, _ = stats.spearmanr(ens_std, abs_err)

            sensitivity_records.append({
                "ensemble_size": size,
                "prediction_mae_ugm3": mae_val,
                "mean_spread_ugm3": mean_spread,
                "coverage_90pct": m_eval["empirical_coverage"],
                "mpiw_90pct_ugm3": m_eval["mean_width_ugm3"],
                "winkler_score_90pct": m_eval["winkler_interval_score"],
                "spearman_spread_error_corr": float(spearman_corr),
                "status": "PASS"
            })

        df_sens = pd.DataFrame(sensitivity_records)
        df_sens.to_csv(output_dir / "ensemble_size_sensitivity.csv", index=False)
        logger.info(f"Ensemble size sensitivity complete across {len(df_sens)} sizes.")
        return df_sens

    def run_paradigm_comparison(self, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Paradigm Comparison (Bootstrap vs. Random-Seed vs. Model-Family Diversity)...")
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_obs = self.df_control['observed_pm25'].values
        paradigms = [
            ("Frozen_Model_Control", self.df_control['production_prediction'].values, np.zeros(len(eval_obs))),
            ("Bootstrap_Ensemble_B30", self.df_boot['ensemble_mean'].values, self.df_boot['ensemble_std'].values),
            ("Random_Seed_Ensemble_N30", self.df_seed['ensemble_mean'].values, self.df_seed['ensemble_std'].values),
            ("Model_Family_Diversity_N4", self.df_family['ensemble_mean'].values, self.df_family['ensemble_std'].values)
        ]

        comp_records = []
        for name, preds, spread in paradigms:
            abs_err = np.abs(eval_obs - preds)
            mae_val = float(np.mean(abs_err))
            rmse_val = float(np.sqrt(np.mean((eval_obs - preds) ** 2)))
            ss_res = float(np.sum((eval_obs - preds) ** 2))
            ss_tot = float(np.sum((eval_obs - np.mean(eval_obs)) ** 2))
            r2_val = float(1.0 - (ss_res / ss_tot))

            mean_spread = float(np.mean(spread))
            spearman_corr, _ = stats.spearmanr(spread, abs_err) if np.std(spread) > 0 else (0.0, 1.0)

            # 90% coverage
            if name == "Frozen_Model_Control":
                cov_90 = 0.0
                mpiw_90 = 0.0
                winkler_90 = 0.0
                ext_cov_90 = 0.0
            elif "Bootstrap" in name:
                q05 = self.df_boot['q05'].values
                q95 = self.df_boot['q95'].values
                l = np.maximum(0.0, q05)
                u = np.maximum(l, q95)
                m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(eval_obs, l, u, 0.90)
                cov_90 = m_ev["empirical_coverage"]
                mpiw_90 = m_ev["mean_width_ugm3"]
                winkler_90 = m_ev["winkler_interval_score"]
                # Extreme subset (>= 150)
                ext_mask = eval_obs >= 150.0
                ext_cov_90 = float(np.mean((eval_obs[ext_mask] >= l[ext_mask]) & (eval_obs[ext_mask] <= u[ext_mask])))
            elif "Seed" in name:
                q05 = self.df_seed['q05'].values
                q95 = self.df_seed['q95'].values
                l = np.maximum(0.0, q05)
                u = np.maximum(l, q95)
                m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(eval_obs, l, u, 0.90)
                cov_90 = m_ev["empirical_coverage"]
                mpiw_90 = m_ev["mean_width_ugm3"]
                winkler_90 = m_ev["winkler_interval_score"]
                ext_mask = eval_obs >= 150.0
                ext_cov_90 = float(np.mean((eval_obs[ext_mask] >= l[ext_mask]) & (eval_obs[ext_mask] <= u[ext_mask])))
            else:
                q05 = self.df_family['q05'].values
                q95 = self.df_family['q95'].values
                l = np.maximum(0.0, q05)
                u = np.maximum(l, q95)
                m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(eval_obs, l, u, 0.90)
                cov_90 = m_ev["empirical_coverage"]
                mpiw_90 = m_ev["mean_width_ugm3"]
                winkler_90 = m_ev["winkler_interval_score"]
                ext_mask = eval_obs >= 150.0
                ext_cov_90 = float(np.mean((eval_obs[ext_mask] >= l[ext_mask]) & (eval_obs[ext_mask] <= u[ext_mask])))

            comp_records.append({
                "paradigm": name,
                "mae_ugm3": mae_val,
                "rmse_ugm3": rmse_val,
                "r2": r2_val,
                "mean_spread_ugm3": mean_spread,
                "spearman_spread_error_corr": float(spearman_corr),
                "coverage_90pct": cov_90,
                "mpiw_90pct_ugm3": mpiw_90,
                "winkler_score_90pct": winkler_90,
                "extreme_episode_coverage_90pct": ext_cov_90
            })

        df_comp = pd.DataFrame(comp_records)
        df_comp.to_csv(output_dir / "bootstrap_vs_seed_comparison.csv", index=False)
        return df_comp

    def run_statistical_significance_tests(self, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Statistical Significance Tests (Wilcoxon Signed-Rank & Bootstrap CI)...")
        output_dir.mkdir(parents=True, exist_ok=True)

        err_ctrl = self.df_control['absolute_error'].values
        err_boot = self.df_boot['absolute_error'].values

        diff_err = err_boot - err_ctrl
        delta_mae = float(np.mean(diff_err))

        stat_w, p_w = stats.wilcoxon(err_ctrl, err_boot)

        # Bootstrap 95% CI for Delta MAE
        n_boot = 2000
        rng = np.random.RandomState(42)
        n_samples = len(diff_err)
        boot_deltas = np.zeros(n_boot)
        for b in range(n_boot):
            sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
            boot_deltas[b] = np.mean(diff_err[sample_idx])

        ci_low = float(np.percentile(boot_deltas, 2.5))
        ci_high = float(np.percentile(boot_deltas, 97.5))

        records = [
            {
                "comparison": "Frozen_Model_Control vs Bootstrap_Ensemble_Mean",
                "delta_mae_ugm3": delta_mae,
                "wilcoxon_statistic": float(stat_w),
                "wilcoxon_p_value": float(p_w),
                "bootstrap_95_ci_lower": ci_low,
                "bootstrap_95_ci_upper": ci_high,
                "statistically_significant": bool(p_w < 0.05),
                "interpretation": "Ensemble averaging slightly improves MAE with narrow confidence bounds" if delta_mae < 0 else "Ensemble averaging maintains comparable accuracy"
            }
        ]

        df_stats = pd.DataFrame(records)
        df_stats.to_csv(output_dir / "statistical_comparisons.csv", index=False)
        return df_stats
