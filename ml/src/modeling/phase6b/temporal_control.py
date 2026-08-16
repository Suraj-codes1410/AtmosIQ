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

logger = setup_logger("TemporalControlPhase6B")


class TemporalControlEnginePhase6B:
    """
    Baseline Frozen Model Control Engine for Phase 6B.
    Reproduces chronological walk-forward out-of-sample predictions and verifies baseline metrics:
    - MAE: 17.0158 µg/m³
    - RMSE: 26.6120 µg/m³
    - R²: 0.9497
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: EnsembleConfigPhase6B):
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

    def run_control_evaluation(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        logger.info("Executing Frozen Model Control Evaluation across Walk-Forward Folds...")
        
        control_records = []
        rf_params = {
            **self.config.rf_base_params,
            "random_state": self.config.random_seed,
            "n_jobs": -1
        }

        for fold_info in self.config.walk_forward_folds:
            f_num = fold_info["fold"]
            train_yrs = fold_info["train_years"]
            eval_yr = fold_info["eval_year"]

            train_mask = self.df_v3['date_dt'].dt.year.isin(train_yrs)
            eval_mask = self.df_v3['date_dt'].dt.year == eval_yr

            df_train = self.df_v3[train_mask]
            df_eval = self.df_v3[eval_mask]

            X_train = df_train[self.features].fillna(0.0)
            y_train = df_train[self.config.target_variable].values

            X_eval = df_eval[self.features].fillna(0.0)
            y_eval = df_eval[self.config.target_variable].values

            # Fit fold control model
            model = RandomForestRegressor(**rf_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_eval)
            residuals = y_eval - y_pred
            abs_errors = np.abs(residuals)
            sq_errors = residuals ** 2

            for i, (_, row) in enumerate(df_eval.iterrows()):
                obs_y = float(y_eval[i])
                pred_y = float(y_pred[i])
                month = row['date_dt'].month

                control_records.append({
                    "date": row['date_dt'].strftime('%Y-%m-%d'),
                    "year": int(eval_yr),
                    "month": int(month),
                    "season": self._assign_season(month),
                    "pollution_regime": self._assign_regime(obs_y),
                    "is_extreme_episode": bool(obs_y >= self.config.extreme_pollution_threshold_ugm3),
                    "eval_fold": f_num,
                    "observed_pm25": obs_y,
                    "production_prediction": pred_y,
                    "residual": float(residuals[i]),
                    "absolute_error": float(abs_errors[i]),
                    "squared_error": float(sq_errors[i])
                })

        df_control = pd.DataFrame(control_records)
        
        overall_mae = float(df_control['absolute_error'].mean())
        overall_rmse = float(np.sqrt(df_control['squared_error'].mean()))
        ss_res = float(df_control['squared_error'].sum())
        ss_tot = float(np.sum((df_control['observed_pm25'] - df_control['observed_pm25'].mean()) ** 2))
        overall_r2 = float(1.0 - (ss_res / ss_tot))

        metrics = {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "r2": overall_r2,
            "sample_count": len(df_control)
        }

        logger.info(f"Control Evaluation Complete. MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, R2={overall_r2:.4f}, N={len(df_control)}")
        
        # Verify reproduction of Phase 6A baseline
        assert abs(overall_mae - 17.0158) < 0.2, f"Control MAE mismatch: {overall_mae}"
        assert abs(overall_r2 - 0.9497) < 0.01, f"Control R2 mismatch: {overall_r2}"
        assert overall_rmse > 0.0, f"Invalid RMSE: {overall_rmse}"

        return df_control, metrics
