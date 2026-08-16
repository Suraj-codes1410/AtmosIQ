"""
AtmosIQ Phase 7C: Machine Learning Utility Assessment (Workstream H).
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


class MLUtilityEvaluator:
    """
    Evaluates downstream forecasting utility on the locked real evaluation fold (2022-2024).
    All models are research artifacts saved under ml/experiments/phase7c/ml_utility/.
    """

    def __init__(self, feature_registry: List[str], model_save_dir: Path, random_seed: int = 42):
        self.feature_registry = list(feature_registry)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed

    def evaluate_ml_utility(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        df_real_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, np.ndarray]]:
        common = [f for f in self.feature_registry if f in df_real_train.columns and f in df_synthetic.columns and f in df_real_test.columns]

        X_real_train = df_real_train[common].values
        y_real_train = df_real_train["pm25"].values

        X_synth = df_synthetic[common].values
        y_synth = df_synthetic["pm25"].values

        X_test = df_real_test[common].values
        y_test = df_real_test["pm25"].values

        configs = {
            "real_only": (X_real_train, y_real_train),
            "synthetic_only": (X_synth, y_synth),
            "real_plus_synthetic_10": (
                np.vstack([X_real_train, X_synth[:int(len(X_synth) * 0.10)]]),
                np.concatenate([y_real_train, y_synth[:int(len(y_synth) * 0.10)]])
            ),
            "real_plus_synthetic_25": (
                np.vstack([X_real_train, X_synth[:int(len(X_synth) * 0.25)]]),
                np.concatenate([y_real_train, y_synth[:int(len(y_synth) * 0.25)]])
            ),
            "real_plus_synthetic_50": (
                np.vstack([X_real_train, X_synth[:int(len(X_synth) * 0.50)]]),
                np.concatenate([y_real_train, y_synth[:int(len(y_synth) * 0.50)]])
            ),
            "real_plus_synthetic_100": (
                np.vstack([X_real_train, X_synth]),
                np.concatenate([y_real_train, y_synth])
            ),
        }

        results = []
        predictions = {}

        for name, (X_tr, y_tr) in configs.items():
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=9,
                min_samples_split=4,
                min_samples_leaf=4,
                random_state=self.random_seed,
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_test)
            predictions[name] = preds

            # Save research model artifact
            model_path = self.model_save_dir / f"model_{name}.joblib"
            joblib.dump(model, model_path)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))
            pr_corr = float(pearsonr(y_test, preds)[0])
            sp_corr = float(spearmanr(y_test, preds)[0])

            results.append({
                "experiment": name,
                "training_samples": len(X_tr),
                "test_samples": len(X_test),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson_r": pr_corr,
                "spearman_rho": sp_corr,
            })

        df_results = pd.DataFrame(results)

        # Baseline comparison
        real_mae = df_results[df_results["experiment"] == "real_only"]["mae"].iloc[0]
        synth_only_mae = df_results[df_results["experiment"] == "synthetic_only"]["mae"].iloc[0]
        aug_100_mae = df_results[df_results["experiment"] == "real_plus_synthetic_100"]["mae"].iloc[0]
        best_aug_mae = df_results[df_results["experiment"].str.startswith("real_plus")]["mae"].min()

        delta_mae_100 = aug_100_mae - real_mae
        delta_best_mae = best_aug_mae - real_mae

        summary = {
            "real_only_mae": real_mae,
            "synthetic_only_mae": synth_only_mae,
            "real_plus_synthetic_100_mae": aug_100_mae,
            "best_augmented_mae": best_aug_mae,
            "delta_mae_augmented_vs_real": delta_mae_100,
            "delta_best_mae_vs_real": delta_best_mae,
            "synthetic_to_real_transfer_pass": (synth_only_mae <= real_mae * 1.35),
            "augmentation_utility_status": "SUPERIOR" if delta_best_mae < -0.2 else ("EQUIVALENT_NON_DEGRADING" if delta_best_mae <= 0.50 else "DEGRADED"),
        }

        return df_results, summary, predictions
