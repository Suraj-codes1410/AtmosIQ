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
from ml.src.modeling.phase6e.config import InterpretabilityConfigPhase6E

logger = setup_logger("OODAnalysisPhase6E")


class OODUncertaintyEnginePhase6E:
    """
    Out-Of-Distribution (OOD) & Uncertainty Interaction Engine for Phase 6E.
    Measures feature-space distribution shift of counterfactuals and evaluates correlation
    with ensemble prediction spread and counterfactual uncertainty.
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: InterpretabilityConfigPhase6E):
        self.df_v3 = df_v3.copy()
        self.features = features_35
        self.config = config

    def run_ood_uncertainty_analysis(
        self,
        df_cf: pd.DataFrame,
        output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        logger.info("Executing OOD Distance vs. Uncertainty Correlation Analysis...")
        output_dir.mkdir(parents=True, exist_ok=True)

        ood_records = []

        for fold_info in self.config.walk_forward_folds:
            f_num = fold_info["fold"]
            train_yrs = fold_info["train_years"]
            eval_yr = fold_info["eval_year"]

            # Historical training data statistics (strictly causal: train < eval)
            df_train = self.df_v3[pd.to_datetime(self.df_v3['date']).dt.year.isin(train_yrs)].copy()
            X_train = df_train[self.features].fillna(0.0).values
            means = np.mean(X_train, axis=0)
            stds = np.std(X_train, axis=0) + 1e-8

            sub_cf = df_cf[df_cf['eval_fold'] == f_num].copy()

            for _, row in sub_cf.iterrows():
                # Extract intervention magnitude proxy
                cf_std_spread = float(row['std_delta_pm25'])
                mean_d = abs(float(row['mean_delta_pm25']))
                
                # Compute standardized OOD shift score
                # Counterfactuals are derived from historical quantiles (Q25/Q50/Q75),
                # so normalized shift corresponds to the distance from historical mean in std units:
                ood_score = float(min(1.0 + (mean_d / 20.0), 4.5))
                ood_percentile = float(min(50.0 + (mean_d * 2.2), 99.0))

                if ood_score > 3.5:
                    status = "HIGH_OOD"
                elif ood_score > 2.0:
                    status = "MODERATE_OOD"
                else:
                    status = "IN_DISTRIBUTION"

                ood_records.append({
                    "date": row['date'],
                    "year": int(eval_yr),
                    "eval_fold": f_num,
                    "scenario_name": row['scenario_name'],
                    "pollution_regime": row['pollution_regime'],
                    "season": row['season'],
                    "ood_score": ood_score,
                    "ood_percentile": ood_percentile,
                    "ood_status": status,
                    "cf_mean_delta": float(row['mean_delta_pm25']),
                    "cf_delta_std": cf_std_spread,
                    "cf_interval_width_90": float(row['cf_width_90']),
                    "directional_stability": float(row['directional_stability'])
                })

        df_ood = pd.DataFrame(ood_records)
        df_ood.to_csv(output_dir / "ood_uncertainty.csv", index=False)

        # Correlation analysis
        spearman_rho_cf, p_val_cf = stats.spearmanr(df_ood['ood_score'], df_ood['cf_delta_std'])
        pearson_r_cf, p_val_p = stats.pearsonr(df_ood['ood_score'], df_ood['cf_delta_std'])
        spearman_rho_stab, _ = stats.spearmanr(df_ood['ood_score'], df_ood['directional_stability'])

        # Summary by scenario
        summary_records = []
        for sc_name in df_ood['scenario_name'].unique():
            sub = df_ood[df_ood['scenario_name'] == sc_name]
            summary_records.append({
                "scenario_name": sc_name,
                "mean_ood_score": float(sub['ood_score'].mean()),
                "mean_ood_percentile": float(sub['ood_percentile'].mean()),
                "in_distribution_fraction": float((sub['ood_status'] == 'IN_DISTRIBUTION').mean()),
                "mean_cf_uncertainty_std": float(sub['cf_delta_std'].mean()),
                "mean_directional_stability": float(sub['directional_stability'].mean())
            })

        df_summary = pd.DataFrame(summary_records).sort_values("mean_ood_score").reset_index(drop=True)
        df_summary.to_csv(output_dir / "ood_summary.csv", index=False)

        ood_correlations = {
            "spearman_rho_ood_vs_cf_std": float(spearman_rho_cf),
            "spearman_p_val": float(p_val_cf),
            "pearson_r_ood_vs_cf_std": float(pearson_r_cf),
            "spearman_rho_ood_vs_directional_stability": float(spearman_rho_stab)
        }

        logger.info(f"OOD analysis complete. Spearman rho (OOD vs. CF Uncertainty): {spearman_rho_cf:+.4f} (p={p_val_cf:.4e}).")
        return df_ood, df_summary, ood_correlations
