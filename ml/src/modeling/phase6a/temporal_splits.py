import sys
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6a.config import UncertaintyConfigPhase6A

logger = setup_logger("TemporalSplitsPhase6A")


class TemporalSplitsEnginePhase6A:
    """
    Temporal Walk-Forward Evaluation and Out-of-Sample Residual Generator for Phase 6A.
    Preserves strict chronological ordering and computes out-of-sample prediction residuals across:
    - Fold 1: Train 2020-2021 (731 days) -> Eval 2022 (365 days)
    - Fold 2: Train 2020-2022 (1096 days) -> Eval 2023 (365 days)
    - Fold 3: Train 2020-2023 (1461 days) -> Eval 2024 (366 days, locked production test fold)
    Total out-of-sample evaluation = 1,096 days.
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: UncertaintyConfigPhase6A):
        self.df_v3 = df_v3.copy()
        self.df_v3['date_dt'] = pd.to_datetime(self.df_v3['date'])
        self.df_v3 = self.df_v3.sort_values('date_dt').reset_index(drop=True)
        self.features = features_35
        self.config = config

    def _assign_season(self, month: int) -> str:
        for s_name, m_list in self.config.seasons.items():
            if month in m_list:
                return s_name
        return "Unknown"

    def _assign_regime(self, pm25_val: float) -> str:
        for r_name, bounds in self.config.pollution_regimes.items():
            if bounds["min"] <= pm25_val < bounds["max"]:
                return r_name
        if pm25_val >= 250.0:
            return "Extreme"
        return "Low"

    def generate_walk_forward_residuals(self, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Generating Walk-Forward Out-of-Sample Predictions & Residuals...")
        output_dir.mkdir(parents=True, exist_ok=True)

        split_summary_records = []
        all_pred_records = []

        rf_params = {
            "n_estimators": 400,
            "max_depth": 9,
            "min_samples_split": 4,
            "min_samples_leaf": 5,
            "max_features": 0.7,
            "random_state": self.config.random_seed,
            "n_jobs": -1
        }

        for fold_info in self.config.walk_forward_folds:
            f_num = fold_info["fold"]
            train_yrs = fold_info["train_years"]
            eval_yr = fold_info["eval_year"]

            train_mask = self.df_v3['date_dt'].dt.year.isin(train_yrs)
            eval_mask = self.df_v3['date_dt'].dt.year == eval_yr

            df_train = self.df_v3[train_mask].copy()
            df_eval = self.df_v3[eval_mask].copy()

            X_train = df_train[self.features].fillna(0.0)
            y_train = df_train[self.config.target_variable].values

            X_eval = df_eval[self.features].fillna(0.0)
            y_eval = df_eval[self.config.target_variable].values

            # Train fold model strictly on historical data
            model_fold = RandomForestRegressor(**rf_params)
            model_fold.fit(X_train, y_train)

            # Predict out-of-sample on evaluation fold
            y_pred = model_fold.predict(X_eval)
            residuals = y_eval - y_pred
            abs_errors = np.abs(residuals)
            sq_errors = residuals ** 2
            rel_errors = np.where(y_eval > 0, abs_errors / y_eval, 0.0)

            mae_fold = float(np.mean(abs_errors))
            rmse_fold = float(np.sqrt(np.mean(sq_errors)))
            r2_fold = float(1.0 - (np.sum(sq_errors) / np.sum((y_eval - np.mean(y_eval)) ** 2)))

            split_summary_records.append({
                "fold": f_num,
                "train_years": f"{min(train_yrs)}-{max(train_yrs)}",
                "train_samples": len(df_train),
                "eval_year": eval_yr,
                "eval_samples": len(df_eval),
                "eval_mae_ugm3": mae_fold,
                "eval_rmse_ugm3": rmse_fold,
                "eval_r2": r2_fold,
                "status": "PASS"
            })

            # Store per-day predictions and residuals
            for i, (_, row) in enumerate(df_eval.iterrows()):
                d_str = row['date_dt'].strftime('%Y-%m-%d')
                obs_y = float(y_eval[i])
                pred_y = float(y_pred[i])
                res_y = float(residuals[i])
                month = row['date_dt'].month
                season = self._assign_season(month)
                regime = self._assign_regime(obs_y)

                all_pred_records.append({
                    "date": d_str,
                    "year": int(row['date_dt'].year),
                    "month": int(month),
                    "day": int(row['date_dt'].day),
                    "season": season,
                    "pollution_regime": regime,
                    "is_extreme_episode": bool(obs_y >= self.config.extreme_pollution_threshold_ugm3),
                    "is_severe_episode": bool(obs_y >= self.config.severe_pollution_threshold_ugm3),
                    "eval_fold": f_num,
                    "observed_pm25": obs_y,
                    "predicted_pm25": pred_y,
                    "residual": res_y,
                    "absolute_error": float(abs_errors[i]),
                    "squared_error": float(sq_errors[i]),
                    "relative_error": float(rel_errors[i])
                })

        df_splits = pd.DataFrame(split_summary_records)
        df_splits.to_csv(output_dir / "temporal_splits.csv", index=False)

        df_preds = pd.DataFrame(all_pred_records)
        df_preds.to_csv(output_dir / "residual_predictions.csv", index=False)

        logger.info(f"Walk-forward residuals generated: {len(df_preds)} total out-of-sample observations.")
        return df_splits, df_preds
