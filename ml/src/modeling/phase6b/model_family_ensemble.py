import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6b.config import EnsembleConfigPhase6B

logger = setup_logger("ModelFamilyEnsemblePhase6B")


class ModelFamilyEnsembleEnginePhase6B:
    """
    Model-Family Diversity Ensemble Engine for Phase 6B.
    Combines distinct tree-based model families (Random Forest, Extra Trees, Gradient Boosting, XGBoost).
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: EnsembleConfigPhase6B):
        self.df_v3 = df_v3.copy()
        self.df_v3['date_dt'] = pd.to_datetime(self.df_v3['date'])
        self.df_v3 = self.df_v3.sort_values('date_dt').reset_index(drop=True)
        self.features = features_35
        self.config = config

    def run_family_ensemble(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Generating Model-Family Diversity Ensemble Predictions...")
        
        family_defs = [
            ("RandomForest", RandomForestRegressor(n_estimators=400, max_depth=9, min_samples_split=4, min_samples_leaf=5, max_features=0.7, random_state=42, n_jobs=-1)),
            ("ExtraTrees", ExtraTreesRegressor(n_estimators=400, max_depth=12, min_samples_split=4, min_samples_leaf=5, max_features=0.7, random_state=42, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)),
            ("XGBoost", xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1))
        ]

        summary_records = []
        interval_records = []
        member_perf_records = []

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
            n_eval = len(df_eval)

            n_models = len(family_defs)
            member_preds = np.zeros((n_models, n_eval))

            for m_idx, (m_name, model_obj) in enumerate(family_defs):
                model_obj.fit(X_train, y_train)
                preds = model_obj.predict(X_eval)
                member_preds[m_idx, :] = preds

                m_mae = float(np.mean(np.abs(y_eval - preds)))
                m_rmse = float(np.sqrt(np.mean((y_eval - preds) ** 2)))
                member_perf_records.append({
                    "eval_fold": f_num,
                    "eval_year": eval_yr,
                    "model_family": m_name,
                    "mae_ugm3": m_mae,
                    "rmse_ugm3": m_rmse
                })

            ens_mean = np.mean(member_preds, axis=0)
            ens_std = np.std(member_preds, axis=0, ddof=1) if n_models > 1 else np.zeros(n_eval)

            q02_5 = np.min(member_preds, axis=0)
            q05 = np.min(member_preds, axis=0)
            q10 = np.min(member_preds, axis=0)
            q90 = np.max(member_preds, axis=0)
            q95 = np.max(member_preds, axis=0)
            q97_5 = np.max(member_preds, axis=0)

            residuals = y_eval - ens_mean
            abs_errors = np.abs(residuals)

            for i, (_, row) in enumerate(df_eval.iterrows()):
                d_str = row['date_dt'].strftime('%Y-%m-%d')
                obs_y = float(y_eval[i])

                summary_records.append({
                    "date": d_str,
                    "eval_fold": f_num,
                    "year": int(eval_yr),
                    "month": int(row['date_dt'].month),
                    "observed_pm25": obs_y,
                    "ensemble_mean": float(ens_mean[i]),
                    "ensemble_std": float(ens_std[i]),
                    "q02_5": float(q02_5[i]),
                    "q05": float(q05[i]),
                    "q10": float(q10[i]),
                    "q90": float(q90[i]),
                    "q95": float(q95[i]),
                    "q97_5": float(q97_5[i]),
                    "residual": float(residuals[i]),
                    "absolute_error": float(abs_errors[i]),
                    "squared_error": float(residuals[i] ** 2),
                    "ensemble_type": "model_family",
                    "ensemble_size": n_models
                })

                for nom_cov, l_raw, u_raw in [
                    (0.80, q10[i], q90[i]),
                    (0.90, q05[i], q95[i]),
                    (0.95, q02_5[i], q97_5[i])
                ]:
                    l_clip = max(0.0, float(l_raw))
                    u_clip = max(l_clip, float(u_raw))
                    interval_records.append({
                        "date": d_str,
                        "eval_fold": f_num,
                        "year": int(eval_yr),
                        "method": "family_clipped",
                        "nominal_coverage": nom_cov,
                        "lower_bound": float(l_clip),
                        "upper_bound": float(u_clip),
                        "interval_width": float(u_clip - l_clip),
                        "observed_pm25": obs_y,
                        "covered": bool(l_clip <= obs_y <= u_clip),
                        "is_clipped": True
                    })

        df_summary = pd.DataFrame(summary_records)
        df_intervals = pd.DataFrame(interval_records)
        df_member_perf = pd.DataFrame(member_perf_records)

        logger.info(f"Model-Family Diversity Complete. Mean Spread = {df_summary['ensemble_std'].mean():.2f} µg/m³.")
        return df_summary, df_intervals, df_member_perf
