import sys
from pathlib import Path
import json
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4h.model_training import ModelFactoryPhase4H

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = setup_logger("OptunaTuningPhase4H")


class OptunaTuningEnginePhase4H:
    """
    Optuna Hyperparameter Optimization Engine for Phase 4H.
    Enforces strict temporal validation inside training years (2020–2023)
    to eliminate future leakage during hyperparameter selection.
    """

    def __init__(self, df_v3: pd.DataFrame):
        self.df = df_v3.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Inner temporal validation split inside training years:
        # Train on 2020-2022, Validate on 2023 (NO exposure to 2024 test)
        self.val_train_mask = self.df['date'].dt.year.isin([2020, 2021, 2022])
        self.val_eval_mask = self.df['date'].dt.year == 2023

    def tune_model(self, model_name: str, feature_set_name: str, features: list, n_trials: int = 25) -> tuple:
        logger.info(f"Starting Optuna tuning for {model_name} on {feature_set_name} ({n_trials} trials)...")

        train_df = self.df[self.val_train_mask]
        eval_df = self.df[self.val_eval_mask]

        X_train = train_df[features].fillna(0.0)
        y_train = train_df['pm25'].values

        X_eval = eval_df[features].fillna(0.0)
        y_eval = eval_df['pm25'].values

        trials_log = []

        def objective(trial: optuna.Trial) -> float:
            if model_name == "Ridge":
                params = {
                    "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)
                }
            elif model_name == "ElasticNet":
                params = {
                    "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95)
                }
            elif model_name == "RandomForest":
                max_feat_idx = trial.suggest_categorical("max_features_choice", ["sqrt", "half", "0.7"])
                max_features = "sqrt" if max_feat_idx == "sqrt" else (0.5 if max_feat_idx == "half" else 0.7)
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 3, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 8),
                    "max_features": max_features
                }
            elif model_name == "XGBoost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 4),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, step=0.01),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.8, step=0.1),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8, step=0.1),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 10.0, step=1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, step=1.0)
                }
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            model = ModelFactoryPhase4H.create_model(model_name, params)
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            val_mae = float(mean_absolute_error(y_eval, preds))

            trials_log.append({
                "trial_number": trial.number,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "hyperparameters": json.dumps(params),
                "validation_score_mae": round(val_mae, 4),
                "objective_function": "inner_temporal_val_mae_2023",
                "random_seed": 42
            })

            return val_mae

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params.copy()
        # Clean up helper keys if present
        if "max_features_choice" in best_params:
            choice = best_params.pop("max_features_choice")
            best_params["max_features"] = "sqrt" if choice == "sqrt" else (0.5 if choice == "half" else 0.7)

        logger.info(f"Optuna Best Trial for {model_name}: MAE={study.best_value:.4f}, Params={best_params}")

        # Mark selected in trials_log
        for entry in trials_log:
            entry["selected"] = (entry["trial_number"] == study.best_trial.number)

        return best_params, trials_log
