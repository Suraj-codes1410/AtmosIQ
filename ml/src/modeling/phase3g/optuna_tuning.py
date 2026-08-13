import sys
import json
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from ml.src.utils.logger import setup_logger

logger = setup_logger("OptunaTuningPhase3G")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuningEnginePhase3G:
    """
    AtmosIQ Phase 3G Optuna Tuning Engine.
    Executes controlled hyperparameter optimization with constrained search spaces using 2-Fold Walk-Forward Development Validation (2022 & 2023).
    The 2024 test set is strictly locked and never used during optimization.
    """

    def __init__(self, modeling_dir: str = "ml/data/modeling/v2", exp_dir: str = "ml/experiments/phase3g"):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.optuna_dir = self.exp_dir / "optuna"
        self.optuna_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.optuna_dir / "optuna.db"

        # Load Dataset v2 frozen snapshot
        self.frozen_file = self.modeling_dir / "feature_dataset_frozen.csv"
        assert self.frozen_file.exists(), f"Frozen dataset missing: {self.frozen_file}"
        self.df = pd.read_csv(self.frozen_file)
        self.df["date_dt"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date_dt").reset_index(drop=True)

        # Development Walk-Forward Folds for Optuna Optimization (MODE A)
        self.dev_folds = [
            {"fold": 1, "train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-12-31"},
            {"fold": 2, "train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-12-31"}
        ]

    def _eval_model_on_folds(self, model_factory_fn, feature_cols: list[str]) -> float:
        """Evaluates a model configuration across Folds 1 & 2 and returns average Validation MAE."""
        maes = []
        for f_info in self.dev_folds:
            tr_sub = self.df[self.df["date_dt"] <= f_info["train_end"]].copy()
            val_sub = self.df[(self.df["date_dt"] >= f_info["val_start"]) & (self.df["date_dt"] <= f_info["val_end"])].copy()

            X_tr = tr_sub[feature_cols]
            y_tr = tr_sub["pm25"]

            X_val = val_sub[feature_cols]
            y_val = val_sub["pm25"]

            model = model_factory_fn()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            mae = float(mean_absolute_error(y_val, preds))
            maes.append(mae)

        return float(np.mean(maes))

    def create_objective(self, model_type: str, feature_cols: list[str]):
        """Returns Optuna objective function for a given model_type and feature set."""

        def objective(trial: optuna.Trial) -> float:
            if model_type == "ridge":
                alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
                factory_fn = lambda: Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=alpha, random_state=42))
                ])

            elif model_type == "elasticnet":
                alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
                factory_fn = lambda: Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42))
                ])

            elif model_type == "random_forest":
                n_estimators = trial.suggest_int("n_estimators", 200, 600, step=50)
                max_depth = trial.suggest_int("max_depth", 4, 10)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 8)
                max_features = trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.75])

                factory_fn = lambda: RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )

            elif model_type == "xgboost":
                n_estimators = trial.suggest_int("n_estimators", 100, 600, step=50)
                max_depth = trial.suggest_int("max_depth", 2, 4)  # Constrained max_depth 2-4
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.05, log=True)
                subsample = trial.suggest_float("subsample", 0.6, 0.85)
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.85)
                min_child_weight = trial.suggest_int("min_child_weight", 3, 10)
                gamma = trial.suggest_float("gamma", 0.0, 2.0)
                reg_alpha = trial.suggest_float("reg_alpha", 1.0, 10.0)
                reg_lambda = trial.suggest_float("reg_lambda", 1.0, 20.0)

                factory_fn = lambda: xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    gamma=gamma,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            return self._eval_model_on_folds(factory_fn, feature_cols)

        return objective

    def run_optuna_studies(self, feature_sets: dict[str, list[str]], target_models: list[str] = None, n_trials: int = 50) -> tuple[dict, pd.DataFrame]:
        """Runs Optuna studies across target models and candidate feature sets."""
        if target_models is None or "all" in target_models:
            models_to_run = ["ridge", "elasticnet", "random_forest", "xgboost"]
        else:
            models_to_run = [m.lower().replace(" ", "_") for m in target_models]

        logger.info(f"Starting Optuna hyperparameter optimization ({n_trials} trials per study)...")
        logger.info(f"Target Models: {models_to_run}")

        best_params_map = {}
        all_trials_records = []
        study_summaries = []

        storage_url = f"sqlite:///{self.db_path}"

        for m_type in models_to_run:
            for fset_name, f_cols in feature_sets.items():
                study_name = f"{m_type}__{fset_name}"
                logger.info(f"Optimizing '{study_name}' ({len(f_cols)} features)...")

                sampler = TPESampler(seed=42)
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    load_if_exists=True,
                    direction="minimize",
                    sampler=sampler
                )

                objective = self.create_objective(m_type, f_cols)
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                best_val_mae = round(float(study.best_value), 4)
                best_params = study.best_params

                logger.info(f"Study Completed: {m_type} on '{fset_name}' | Best Dev Mean MAE: {best_val_mae}")

                best_params_map[study_name] = {
                    "model_type": m_type,
                    "feature_set": fset_name,
                    "feature_count": len(f_cols),
                    "best_dev_mean_mae": best_val_mae,
                    "params": best_params
                }

                study_summaries.append({
                    "study_name": study_name,
                    "model_type": m_type,
                    "feature_set": fset_name,
                    "feature_count": len(f_cols),
                    "best_dev_mean_mae": best_val_mae,
                    "best_trial_id": study.best_trial.number
                })

                for t in study.trials:
                    all_trials_records.append({
                        "study_name": study_name,
                        "model_type": m_type,
                        "feature_set": fset_name,
                        "trial_id": t.number,
                        "val_mae": round(float(t.value), 4) if t.value is not None else None,
                        "params": json.dumps(t.params),
                        "state": t.state.name
                    })

        # Save artifacts under ml/experiments/phase3g/optuna/
        with open(self.optuna_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params_map, f, indent=4)

        trials_df = pd.DataFrame(all_trials_records)
        trials_df.to_csv(self.optuna_dir / "trials.csv", index=False)
        trials_df.to_csv(self.optuna_dir / "optuna_history.csv", index=False)

        summary_df = pd.DataFrame(study_summaries).sort_values("best_dev_mean_mae").reset_index(drop=True)
        summary_df.to_csv(self.optuna_dir / "study_summary.csv", index=False)

        logger.info(f"Optuna optimization completed. Results persisted to: {self.optuna_dir}")
        return best_params_map, summary_df


if __name__ == "__main__":
    from ml.src.modeling.phase3g.feature_sets import FeatureSetManagerPhase3G
    fsets = FeatureSetManagerPhase3G().get_phase3g_feature_sets()
    engine = OptunaTuningEnginePhase3G()
    engine.run_optuna_studies(fsets, n_trials=5)
