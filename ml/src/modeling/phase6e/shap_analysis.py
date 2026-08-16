import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6e.config import InterpretabilityConfigPhase6E

logger = setup_logger("SHAPAnalysisPhase6E")


class SHAPUncertaintyEnginePhase6E:
    """
    Feature-Level TreeSHAP Attribution Uncertainty Engine for Phase 6E.
    Computes TreeSHAP distributions across B=30 bootstrap ensemble members per walk-forward fold,
    evaluating attribution intervals, sign stability, and additivity.
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: InterpretabilityConfigPhase6E):
        self.df_v3 = df_v3.copy()
        self.df_v3['date_dt'] = pd.to_datetime(self.df_v3['date'])
        self.df_v3 = self.df_v3.sort_values('date_dt').reset_index(drop=True)
        
        # Ensure season and pollution_regime exist
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

    def run_shap_ensemble_analysis(
        self,
        output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        logger.info(f"Executing TreeSHAP Ensemble Attribution Uncertainty Analysis (B={self.config.ensemble_size_B})...")
        output_dir.mkdir(parents=True, exist_ok=True)

        B = self.config.ensemble_size_B
        rf_base = {
            **self.config.rf_base_params,
            "n_jobs": -1
        }

        all_obs_records = []
        all_models_by_fold = {}
        additivity_failures = 0
        total_evaluations = 0

        for fold_info in self.config.walk_forward_folds:
            f_num = fold_info["fold"]
            train_yrs = fold_info["train_years"]
            eval_yr = fold_info["eval_year"]

            train_mask = self.df_v3['date_dt'].dt.year.isin(train_yrs)
            eval_mask = self.df_v3['date_dt'].dt.year == eval_yr

            df_train = self.df_v3[train_mask].reset_index(drop=True)
            df_eval = self.df_v3[eval_mask].reset_index(drop=True)

            X_train = df_train[self.features].fillna(0.0).values
            y_train = df_train[self.config.target_variable].values
            X_eval = df_eval[self.features].fillna(0.0).values
            y_eval = df_eval[self.config.target_variable].values

            n_train = len(df_train)
            n_eval = len(df_eval)

            logger.info(f"Fold {f_num} (Eval Year {eval_yr}): Training B={B} bootstrap models on {n_train} samples...")

            fold_models = []
            fold_explainers = []

            for b in range(B):
                rng = np.random.RandomState(self.config.random_seed + f_num * 1000 + b)
                boot_idx = rng.choice(n_train, size=n_train, replace=True)
                X_boot = X_train[boot_idx]
                y_boot = y_train[boot_idx]

                model = RandomForestRegressor(**rf_base, random_state=self.config.random_seed + b)
                model.fit(X_boot, y_boot)
                fold_models.append(model)

                # TreeSHAP Explainer
                explainer = shap.TreeExplainer(model)
                fold_explainers.append(explainer)

            all_models_by_fold[f_num] = {
                "models": fold_models,
                "explainers": fold_explainers,
                "df_eval": df_eval,
                "X_eval": X_eval,
                "y_eval": y_eval
            }

            # Compute SHAP matrix across all members: shape (B, n_eval, n_features)
            shap_ensemble = np.zeros((B, n_eval, len(self.features)))
            preds_ensemble = np.zeros((B, n_eval))
            expected_values = np.zeros(B)

            for b in range(B):
                expl = fold_explainers[b]
                model = fold_models[b]
                sv = expl.shap_values(X_eval, check_additivity=False)
                shap_ensemble[b] = sv
                preds_ensemble[b] = model.predict(X_eval)
                ev = expl.expected_value
                expected_values[b] = ev[0] if isinstance(ev, (list, np.ndarray)) else ev

            # For each evaluation observation, summarize feature attribution distributions
            for i in range(n_eval):
                row = df_eval.iloc[i]
                d_str = row['date']
                obs_y = float(y_eval[i])
                total_evaluations += 1

                for j, feat_name in enumerate(self.features):
                    f_shaps = shap_ensemble[:, i, j]  # B values
                    m_s = float(np.mean(f_shaps))
                    med_s = float(np.median(f_shaps))
                    std_s = float(np.std(f_shaps, ddof=1))
                    q05_s = float(np.percentile(f_shaps, 5))
                    q10_s = float(np.percentile(f_shaps, 10))
                    q25_s = float(np.percentile(f_shaps, 25))
                    q75_s = float(np.percentile(f_shaps, 75))
                    q90_s = float(np.percentile(f_shaps, 90))
                    q95_s = float(np.percentile(f_shaps, 95))

                    # Sign stability calculation
                    eps = self.config.near_zero_epsilon
                    pos_count = np.sum(f_shaps > eps)
                    neg_count = np.sum(f_shaps < -eps)
                    near_zero_count = np.sum(np.abs(f_shaps) <= eps)

                    pos_frac = float(pos_count / B)
                    neg_frac = float(neg_count / B)
                    near_zero_frac = float(near_zero_count / B)

                    # Directional sign stability
                    if near_zero_frac >= 0.80:
                        sign_stab = 1.0 - near_zero_frac  # Mostly neutral
                        dominant_dir = "NEUTRAL"
                    elif pos_frac >= neg_frac:
                        sign_stab = pos_frac
                        dominant_dir = "POSITIVE"
                    else:
                        sign_stab = neg_frac
                        dominant_dir = "NEGATIVE"

                    all_obs_records.append({
                        "date": d_str,
                        "year": int(eval_yr),
                        "eval_fold": f_num,
                        "pollution_regime": row['pollution_regime'],
                        "season": row['season'],
                        "feature_name": feat_name,
                        "observed_pm25": obs_y,
                        "ensemble_mean_prediction": float(np.mean(preds_ensemble[:, i])),
                        "mean_shap": m_s,
                        "median_shap": med_s,
                        "std_shap": std_s,
                        "q05_shap": q05_s,
                        "q10_shap": q10_s,
                        "q25_shap": q25_s,
                        "q75_shap": q75_s,
                        "q90_shap": q90_s,
                        "q95_shap": q95_s,
                        "attr_interval_80_lower": q10_s,
                        "attr_interval_80_upper": q90_s,
                        "attr_interval_90_lower": q05_s,
                        "attr_interval_90_upper": q95_s,
                        "attr_width_90": q95_s - q05_s,
                        "sign_stability": sign_stab,
                        "dominant_direction": dominant_dir,
                        "positive_fraction": pos_frac,
                        "negative_fraction": neg_frac,
                        "near_zero_fraction": near_zero_frac
                    })

                # Check additivity for ensemble mean
                mean_ev = np.mean(expected_values)
                sum_mean_shaps = np.sum([np.mean(shap_ensemble[:, i, j]) for j in range(len(self.features))])
                reconstructed_pred = mean_ev + sum_mean_shaps
                pred_error = abs(reconstructed_pred - np.mean(preds_ensemble[:, i]))
                if pred_error > self.config.additivity_tolerance:
                    additivity_failures += 1

        df_shap_obs = pd.DataFrame(all_obs_records)
        df_shap_obs.to_csv(output_dir / "shap_uncertainty.csv", index=False)

        # Global feature-level summary
        feat_summaries = []
        for feat_name in self.features:
            sub = df_shap_obs[df_shap_obs['feature_name'] == feat_name]
            mean_abs_mag = float(sub['mean_shap'].abs().mean())
            mean_signed = float(sub['mean_shap'].mean())
            mean_std = float(sub['std_shap'].mean())
            mean_sign_stab = float(sub['sign_stability'].mean())
            pos_frac_avg = float(sub['positive_fraction'].mean())
            neg_frac_avg = float(sub['negative_fraction'].mean())

            if mean_sign_stab >= self.config.high_stability_threshold:
                stab_class = "HIGH_STABILITY"
            elif mean_sign_stab >= self.config.moderate_stability_threshold:
                stab_class = "MODERATE_STABILITY"
            else:
                stab_class = "LOW_STABILITY"

            feat_summaries.append({
                "feature_name": feat_name,
                "mean_absolute_shap": mean_abs_mag,
                "mean_signed_shap": mean_signed,
                "mean_shap_std": mean_std,
                "mean_sign_stability": mean_sign_stab,
                "stability_classification": stab_class,
                "average_positive_fraction": pos_frac_avg,
                "average_negative_fraction": neg_frac_avg,
                "importance_rank": 0  # assigned below
            })

        df_feat_summary = pd.DataFrame(feat_summaries).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
        df_feat_summary["importance_rank"] = np.arange(1, len(df_feat_summary) + 1)
        df_feat_summary.to_csv(output_dir / "shap_feature_summary.csv", index=False)

        # Sign stability summary
        df_sign_stab = df_feat_summary[[
            "feature_name", "importance_rank", "mean_absolute_shap", "mean_sign_stability",
            "stability_classification", "average_positive_fraction", "average_negative_fraction"
        ]].copy()
        df_sign_stab.to_csv(output_dir / "shap_sign_stability.csv", index=False)

        shap_diagnostics = {
            "total_evaluations": total_evaluations,
            "additivity_failures": additivity_failures,
            "additivity_pass_rate": float(1.0 - (additivity_failures / max(total_evaluations, 1))),
            "top_feature_by_importance": df_feat_summary.iloc[0]["feature_name"],
            "top_feature_mean_abs_shap": float(df_feat_summary.iloc[0]["mean_absolute_shap"]),
            "high_stability_feature_count": int((df_feat_summary["stability_classification"] == "HIGH_STABILITY").sum()),
            "moderate_stability_feature_count": int((df_feat_summary["stability_classification"] == "MODERATE_STABILITY").sum()),
            "low_stability_feature_count": int((df_feat_summary["stability_classification"] == "LOW_STABILITY").sum())
        }

        logger.info(f"TreeSHAP analysis complete. High stability features: {shap_diagnostics['high_stability_feature_count']}/35, Additivity pass rate: {shap_diagnostics['additivity_pass_rate']*100:.2f}%")
        return df_shap_obs, df_feat_summary, df_sign_stab, all_models_by_fold, shap_diagnostics
