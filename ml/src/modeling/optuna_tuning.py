import sys
import json
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import optuna

from ml.src.utils.logger import setup_logger

logger = setup_logger("OptunaTuningPhase3D")

# Suppress Optuna info log flood
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


class OptunaTuningEngine:
    """
    AtmosIQ Phase 3C & 3D: Optuna Hyperparameter Optimization Engine.
    Tunes Ridge, ElasticNet, Random Forest, and XGBoost using Validation MAE as objective.
    Strict zero test set leakage!
    """

    def __init__(
        self,
        modeling_dir: str = "ml/data/modeling/v1",
        exp_dir: str = "ml/experiments/phase3d"
    ):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.best_params_dir = self.exp_dir / "best_parameters"
        self.metrics_dir = self.exp_dir / "metrics"
        self.best_params_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = "pm25"
        self.date_col = "date"
        self.n_trials = 50

    def load_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads train, validation, and test split DataFrames."""
        tr_df = pd.read_csv(self.modeling_dir / "train.csv")
        val_df = pd.read_csv(self.modeling_dir / "validation.csv")
        te_df = pd.read_csv(self.modeling_dir / "test.csv")
        return tr_df, val_df, te_df

    def optimize_study(
        self,
        model_name: str,
        feature_set_name: str,
        f_cols: list[str],
        tr_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> tuple[dict, float, pd.DataFrame]:
        """Runs Optuna study for a specific model family and feature set."""
        X_tr = tr_df[f_cols].copy()
        y_tr = tr_df[self.target_col].copy()

        X_val = val_df[f_cols].copy()
        y_val = val_df[self.target_col].copy()

        trial_logs = []

        def objective(trial: optuna.Trial) -> float:
            if model_name == "Ridge":
                alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=alpha, random_state=42))
                ])
                pipeline.fit(X_tr, y_tr)
                val_preds = pipeline.predict(X_val)

            elif model_name == "ElasticNet":
                alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42))
                ])
                pipeline.fit(X_tr, y_tr)
                val_preds = pipeline.predict(X_val)

            elif model_name == "Random Forest":
                n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)
                max_depth = trial.suggest_int("max_depth", 4, 8)
                min_samples_split = trial.suggest_int("min_samples_split", 5, 20)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 8)
                max_feat_type = trial.suggest_categorical("max_features_type", ["sqrt", "float"])
                max_features = "sqrt" if max_feat_type == "sqrt" else trial.suggest_float("max_features_float", 0.4, 0.8)

                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_tr, y_tr)
                val_preds = model.predict(X_val)

            elif model_name == "XGBoost":
                n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
                max_depth = trial.suggest_int("max_depth", 2, 4)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.05, log=True)
                min_child_weight = trial.suggest_int("min_child_weight", 3, 15)
                subsample = trial.suggest_float("subsample", 0.6, 0.9)
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.8)
                gamma = trial.suggest_float("gamma", 0.0, 2.0)
                reg_alpha = trial.suggest_float("reg_alpha", 1.0, 20.0)
                reg_lambda = trial.suggest_float("reg_lambda", 1.0, 20.0)

                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    gamma=gamma,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_tr, y_tr)
                val_preds = model.predict(X_val)

            else:
                raise ValueError(f"Unknown model family: {model_name}")

            val_mae = float(mean_absolute_error(y_val, val_preds))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            val_r2 = float(r2_score(y_val, val_preds))

            trial_logs.append({
                "trial_number": trial.number,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "val_mae": round(val_mae, 4),
                "val_rmse": round(val_rmse, 4),
                "val_r2": round(val_r2, 4),
                "params": str(trial.params)
            })

            return val_mae

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_val_mae = study.best_value
        study_df = pd.DataFrame(trial_logs)

        logger.info(f"Study Completed: {model_name} on '{feature_set_name}' | Best Val MAE: {best_val_mae:.4f}")
        return best_params, best_val_mae, study_df

    def run_all_studies(self, feature_sets: dict[str, list[str]]) -> tuple[dict, pd.DataFrame]:
        """Runs Optuna studies for 4 model families across all Phase 3D feature sets."""
        tr_df, val_df, te_df = self.load_splits()
        models = ["Ridge", "ElasticNet", "Random Forest", "XGBoost"]

        all_best_params = {}
        all_trials_list = []

        for fset_name, f_cols in feature_sets.items():
            for m_name in models:
                key = f"{m_name.lower().replace(' ', '_')}__{fset_name}"
                logger.info(f"Starting Optuna Optimization for '{key}' ({len(f_cols)} features)...")

                best_params, best_mae, study_df = self.optimize_study(m_name, fset_name, f_cols, tr_df, val_df)
                all_best_params[key] = {
                    "model_name": m_name,
                    "feature_set": fset_name,
                    "feature_count": len(f_cols),
                    "best_val_mae": round(best_mae, 4),
                    "best_params": best_params
                }
                all_trials_list.append(study_df)

        # Save best_params.json
        best_params_file = self.best_params_dir / "best_params.json"
        with open(best_params_file, "w", encoding="utf-8") as f:
            json.dump(all_best_params, f, indent=4)
        logger.info(f"Best parameters saved to: {best_params_file}")

        # Save optimization_results.csv
        combined_trials_df = pd.concat(all_trials_list, ignore_index=True)
        trials_file = self.metrics_dir / "optimization_results.csv"
        combined_trials_df.to_csv(trials_file, index=False)
        logger.info(f"Optimization trial history saved to: {trials_file}")

        return all_best_params, combined_trials_df


if __name__ == "__main__":
    from ml.src.modeling.feature_sets import FeatureSetManager
    mgr = FeatureSetManager()
    fsets = mgr.get_phase3d_feature_sets()

    engine = OptunaTuningEngine()
    engine.run_all_studies(fsets)
