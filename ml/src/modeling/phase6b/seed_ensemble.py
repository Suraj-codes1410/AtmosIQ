import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6b.config import EnsembleConfigPhase6B

logger = setup_logger("SeedEnsemblePhase6B")


class SeedEnsembleEnginePhase6B:
    """
    Random-Seed Ensemble Uncertainty Engine for Phase 6B.
    Constructs an ensemble of N models trained on identical historical training data with varying random seeds (0 to N-1).
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: EnsembleConfigPhase6B):
        self.df_v3 = df_v3.copy()
        self.df_v3['date_dt'] = pd.to_datetime(self.df_v3['date'])
        self.df_v3 = self.df_v3.sort_values('date_dt').reset_index(drop=True)
        self.features = features_35
        self.config = config

    def run_seed_ensemble(self, N: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        logger.info(f"Generating Random-Seed Ensemble Predictions (N={N})...")
        
        summary_records = []
        interval_records = []
        all_member_preds = []

        rf_base = {
            **self.config.rf_base_params,
            "n_jobs": -1
        }

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

            fold_member_preds = np.zeros((N, n_eval))

            for s_idx in range(N):
                seed_val = self.config.random_seed * 100 + s_idx
                model = RandomForestRegressor(**rf_base, random_state=seed_val)
                model.fit(X_train, y_train)
                fold_member_preds[s_idx, :] = model.predict(X_eval)

            all_member_preds.append(fold_member_preds)

            ens_mean = np.mean(fold_member_preds, axis=0)
            ens_std = np.std(fold_member_preds, axis=0, ddof=1) if N > 1 else np.zeros(n_eval)

            q02_5 = np.percentile(fold_member_preds, 2.5, axis=0)
            q05 = np.percentile(fold_member_preds, 5.0, axis=0)
            q10 = np.percentile(fold_member_preds, 10.0, axis=0)
            q90 = np.percentile(fold_member_preds, 90.0, axis=0)
            q95 = np.percentile(fold_member_preds, 95.0, axis=0)
            q97_5 = np.percentile(fold_member_preds, 97.5, axis=0)

            residuals = y_eval - ens_mean
            abs_errors = np.abs(residuals)

            for i, (_, row) in enumerate(df_eval.iterrows()):
                d_str = row['date_dt'].strftime('%Y-%m-%d')
                obs_y = float(y_eval[i])
                m_y = float(ens_mean[i])
                s_y = float(ens_std[i])

                summary_records.append({
                    "date": d_str,
                    "eval_fold": f_num,
                    "year": int(eval_yr),
                    "month": int(row['date_dt'].month),
                    "observed_pm25": obs_y,
                    "ensemble_mean": m_y,
                    "ensemble_std": s_y,
                    "q02_5": float(q02_5[i]),
                    "q05": float(q05[i]),
                    "q10": float(q10[i]),
                    "q90": float(q90[i]),
                    "q95": float(q95[i]),
                    "q97_5": float(q97_5[i]),
                    "residual": float(residuals[i]),
                    "absolute_error": float(abs_errors[i]),
                    "squared_error": float(residuals[i] ** 2),
                    "ensemble_type": "seed",
                    "ensemble_size": N
                })

                for nom_cov, l_raw, u_raw in [
                    (0.80, q10[i], q90[i]),
                    (0.90, q05[i], q95[i]),
                    (0.95, q02_5[i], q97_5[i])
                ]:
                    # Raw
                    interval_records.append({
                        "date": d_str,
                        "eval_fold": f_num,
                        "year": int(eval_yr),
                        "method": "seed_raw",
                        "nominal_coverage": nom_cov,
                        "lower_bound": float(l_raw),
                        "upper_bound": float(u_raw),
                        "interval_width": float(u_raw - l_raw),
                        "observed_pm25": obs_y,
                        "covered": bool(l_raw <= obs_y <= u_raw),
                        "is_clipped": False
                    })
                    # Physically Clipped
                    l_clip = max(0.0, float(l_raw))
                    u_clip = max(l_clip, float(u_raw))
                    interval_records.append({
                        "date": d_str,
                        "eval_fold": f_num,
                        "year": int(eval_yr),
                        "method": "seed_clipped",
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

        logger.info(f"Random-Seed Ensemble Complete (N={N}). N={len(df_summary)}, Mean Spread={df_summary['ensemble_std'].mean():.2f} µg/m³.")
        return df_summary, df_intervals, all_member_preds
