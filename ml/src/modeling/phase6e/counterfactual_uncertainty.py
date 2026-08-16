import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6e.config import InterpretabilityConfigPhase6E
from ml.src.modeling.phase6e.group_attribution import GroupAttributionEnginePhase6E

logger = setup_logger("CounterfactualUncertaintyPhase6E")


class CounterfactualUncertaintyEnginePhase6E:
    """
    Counterfactual Uncertainty & Directional Stability Engine for Phase 6E.
    Applies predefined environmental scenarios to held-out observations and measures
    the dispersion and directional stability of ΔPM2.5 predictions across B=30 bootstrap ensemble members.
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: InterpretabilityConfigPhase6E):
        self.df_v3 = df_v3.copy()
        self.df_v3['date_dt'] = pd.to_datetime(self.df_v3['date'])
        self.df_v3 = self.df_v3.sort_values('date_dt').reset_index(drop=True)

        if 'season' not in self.df_v3.columns:
            m = self.df_v3['date_dt'].dt.month
            self.df_v3['season'] = np.where(m.isin([12, 1, 2]), "Winter",
                                   np.where(m.isin([3, 4, 5]), "Summer",
                                   np.where(m.isin([6, 7, 8, 9]), "Monsoon", "Post-Monsoon")))

        if 'pollution_regime' not in self.df_v3.columns:
            p = self.df_v3[config.target_variable]
            self.df_v3['pollution_regime'] = np.where(p < 60.0, "Low",
                                             np.where(p < 120.0, "Moderate",
                                             np.where(p < 250.0, "High", "Extreme")))

        self.features = features_35
        self.config = config
        self.group_defs = GroupAttributionEnginePhase6E.GROUP_DEFINITIONS

    def run_counterfactual_uncertainty_analysis(
        self,
        all_models_by_fold: Dict[int, Dict[str, Any]],
        output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Executing Counterfactual Uncertainty & Directional Stability Analysis across Ensemble Models...")
        output_dir.mkdir(parents=True, exist_ok=True)

        cf_records = []
        B = self.config.ensemble_size_B

        for fold_info in self.config.walk_forward_folds:
            f_num = fold_info["fold"]
            train_yrs = fold_info["train_years"]
            eval_yr = fold_info["eval_year"]

            # Historical training data reference quantiles (strict causality: train < eval)
            df_train = self.df_v3[pd.to_datetime(self.df_v3['date']).dt.year.isin(train_yrs)].copy()
            fold_data = all_models_by_fold[f_num]
            models = fold_data["models"]
            df_eval = fold_data["df_eval"]
            X_eval = fold_data["X_eval"]

            # Calculate historical feature quantiles
            q25_dict = {feat: float(df_train[feat].quantile(0.25)) for feat in self.features}
            q50_dict = {feat: float(df_train[feat].quantile(0.50)) for feat in self.features}
            q75_dict = {feat: float(df_train[feat].quantile(0.75)) for feat in self.features}

            # Evaluate counterfactual scenarios
            for sc_info in self.config.counterfactual_scenarios:
                sc_name = sc_info["scenario_name"]
                
                # Determine feature interventions
                feature_overrides = {}
                if sc_name == "biomass_low":
                    for f in self.group_defs["biomass_burning"]:
                        feature_overrides[f] = q25_dict[f]
                elif sc_name == "biomass_median":
                    for f in self.group_defs["biomass_burning"]:
                        feature_overrides[f] = q50_dict[f]
                elif sc_name == "biomass_high":
                    for f in self.group_defs["biomass_burning"]:
                        feature_overrides[f] = q75_dict[f]
                elif sc_name == "wind_stagnant":
                    for f in self.group_defs["wind_ventilation"]:
                        feature_overrides[f] = q25_dict[f]
                elif sc_name == "wind_dispersion":
                    for f in self.group_defs["wind_ventilation"]:
                        feature_overrides[f] = q75_dict[f]
                elif sc_name == "meteorology_normal":
                    for f in self.group_defs["meteorology"]:
                        feature_overrides[f] = q50_dict[f]
                elif sc_name == "combined_biomass_wind":
                    for f in self.group_defs["biomass_burning"]:
                        feature_overrides[f] = q25_dict[f]
                    for f in self.group_defs["wind_ventilation"]:
                        feature_overrides[f] = q75_dict[f]
                elif sc_name == "combined_all_favorable":
                    for f in self.group_defs["biomass_burning"]:
                        feature_overrides[f] = q25_dict[f]
                    for f in self.group_defs["wind_ventilation"]:
                        feature_overrides[f] = q75_dict[f]
                    for f in self.group_defs["meteorology"]:
                        feature_overrides[f] = q50_dict[f]

                # Create perturbed evaluation matrix
                X_cf = X_eval.copy()
                for f_name, val in feature_overrides.items():
                    if f_name in self.features:
                        idx = self.features.index(f_name)
                        X_cf[:, idx] = val

                # Evaluate all B models on original and counterfactual matrices
                preds_orig = np.zeros((B, len(df_eval)))
                preds_cf = np.zeros((B, len(df_eval)))

                for b in range(B):
                    preds_orig[b] = models[b].predict(X_eval)
                    preds_cf[b] = models[b].predict(X_cf)

                # Physical non-negativity constraint
                preds_orig = np.maximum(0.0, preds_orig)
                preds_cf = np.maximum(0.0, preds_cf)

                delta_matrix = preds_cf - preds_orig  # shape (B, n_eval)

                # Summarize per evaluation observation
                for i in range(len(df_eval)):
                    row = df_eval.iloc[i]
                    deltas = delta_matrix[:, i]  # B values

                    mean_d = float(np.mean(deltas))
                    med_d = float(np.median(deltas))
                    std_d = float(np.std(deltas, ddof=1))
                    q05_d = float(np.percentile(deltas, 5))
                    q10_d = float(np.percentile(deltas, 10))
                    q25_d = float(np.percentile(deltas, 25))
                    q75_d = float(np.percentile(deltas, 75))
                    q90_d = float(np.percentile(deltas, 90))
                    q95_d = float(np.percentile(deltas, 95))

                    # Directional stability calculation
                    if abs(mean_d) < 0.10:
                        dir_stab = 1.0  # Stable neutral
                    elif mean_d > 0:
                        dir_stab = float(np.mean(deltas > 0))
                    else:
                        dir_stab = float(np.mean(deltas < 0))

                    cf_records.append({
                        "date": row['date'],
                        "year": int(eval_yr),
                        "eval_fold": f_num,
                        "pollution_regime": row['pollution_regime'],
                        "season": row['season'],
                        "scenario_name": sc_name,
                        "observed_pm25": float(row['pm25']),
                        "baseline_ensemble_mean": float(np.mean(preds_orig[:, i])),
                        "cf_ensemble_mean": float(np.mean(preds_cf[:, i])),
                        "mean_delta_pm25": mean_d,
                        "median_delta_pm25": med_d,
                        "std_delta_pm25": std_d,
                        "q05_delta_pm25": q05_d,
                        "q10_delta_pm25": q10_d,
                        "q25_delta_pm25": q25_d,
                        "q75_delta_pm25": q75_d,
                        "q90_delta_pm25": q90_d,
                        "q95_delta_pm25": q95_d,
                        "cf_interval_80_lower": q10_d,
                        "cf_interval_80_upper": q90_d,
                        "cf_interval_90_lower": q05_d,
                        "cf_interval_90_upper": q95_d,
                        "cf_width_90": q95_d - q05_d,
                        "directional_stability": dir_stab,
                        "physical_validity_status": "VALID"
                    })

        df_cf = pd.DataFrame(cf_records)
        df_cf.to_csv(output_dir / "counterfactual_uncertainty.csv", index=False)

        # Scenario summary table
        sc_summary_records = []
        for sc_name in df_cf['scenario_name'].unique():
            sub = df_cf[df_cf['scenario_name'] == sc_name]
            sc_summary_records.append({
                "scenario_name": sc_name,
                "observation_count": len(sub),
                "mean_delta_pm25": float(sub['mean_delta_pm25'].mean()),
                "median_delta_pm25": float(sub['median_delta_pm25'].mean()),
                "mean_delta_std": float(sub['std_delta_pm25'].mean()),
                "mean_directional_stability": float(sub['directional_stability'].mean()),
                "q10_delta_mean": float(sub['q10_delta_pm25'].mean()),
                "q90_delta_mean": float(sub['q90_delta_pm25'].mean()),
                "mean_interval_width_90": float(sub['cf_width_90'].mean())
            })

        df_sc_summary = pd.DataFrame(sc_summary_records).sort_values("mean_delta_pm25").reset_index(drop=True)
        df_sc_summary.to_csv(output_dir / "counterfactual_cases.csv", index=False)
        logger.info(f"Counterfactual uncertainty complete across {len(df_sc_summary)} scenarios.")
        return df_cf, df_sc_summary
