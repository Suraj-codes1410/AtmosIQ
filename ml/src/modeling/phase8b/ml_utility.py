"""
AtmosIQ Phase 8B: Downstream Machine Learning Scaling Utility Evaluator.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


class MLUtilityScaleEvaluator:
    """Evaluates ML utility across progressive synthetic scaling augmentation ratios."""

    def __init__(self, feature_registry: List[str], random_seed: int = 42):
        self.feature_registry = list(feature_registry)
        self.random_seed = random_seed

    def evaluate_scaling_utility(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic_corpus: pd.DataFrame,
        df_real_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        common = [f for f in self.feature_registry if f in df_real_train.columns and f in df_synthetic_corpus.columns and f in df_real_test.columns]

        X_real_train = df_real_train[common].values
        y_real_train = df_real_train["pm25"].values

        X_synth = df_synthetic_corpus[common].values
        y_synth = df_synthetic_corpus["pm25"].values

        X_test = df_real_test[common].values
        y_test = df_real_test["pm25"].values

        ratios = {
            "real_only": (X_real_train, y_real_train),
            "real_plus_10pct": (
                np.vstack([X_real_train, X_synth[:int(len(X_synth) * 0.10)]]) if len(X_synth) > 0 else X_real_train,
                np.concatenate([y_real_train, y_synth[:int(len(y_synth) * 0.10)]]) if len(y_synth) > 0 else y_real_train
            ),
            "real_plus_25pct": (
                np.vstack([X_real_train, X_synth[:int(len(X_synth) * 0.25)]]) if len(X_synth) > 0 else X_real_train,
                np.concatenate([y_real_train, y_synth[:int(len(y_synth) * 0.25)]]) if len(y_synth) > 0 else y_real_train
            ),
            "real_plus_50pct": (
                np.vstack([X_real_train, X_synth[:int(len(X_synth) * 0.50)]]) if len(X_synth) > 0 else X_real_train,
                np.concatenate([y_real_train, y_synth[:int(len(y_synth) * 0.50)]]) if len(y_synth) > 0 else y_real_train
            ),
            "real_plus_full_scaled": (
                np.vstack([X_real_train, X_synth]) if len(X_synth) > 0 else X_real_train,
                np.concatenate([y_real_train, y_synth]) if len(y_synth) > 0 else y_real_train
            ),
        }

        records = []
        for name, (X_tr, y_tr) in ratios.items():
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=9,
                min_samples_split=4,
                min_samples_leaf=4,
                random_state=self.random_seed,
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_test)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))
            pr_corr = float(pearsonr(y_test, preds)[0])

            # Extreme 250+ MAE
            ext_mask = (y_test >= 250.0)
            ext_mae = float(mean_absolute_error(y_test[ext_mask], preds[ext_mask])) if ext_mask.any() else 0.0

            records.append({
                "augmentation_configuration": name,
                "training_sample_count": len(X_tr),
                "synthetic_sample_count": len(X_tr) - len(X_real_train),
                "test_sample_count": len(X_test),
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
                "pearson_r": pr_corr,
                "extreme_250_mae": ext_mae,
            })

        df_ml = pd.DataFrame(records)
        real_mae = df_ml[df_ml["augmentation_configuration"] == "real_only"]["test_mae"].iloc[0]
        best_aug_mae = df_ml[df_ml["augmentation_configuration"] != "real_only"]["test_mae"].min()

        summary = {
            "real_only_mae": real_mae,
            "best_augmented_mae": best_aug_mae,
            "delta_best_mae": best_aug_mae - real_mae,
            "ml_scaling_status": "PASS" if (best_aug_mae - real_mae) <= 0.50 else "WARNING",
        }

        return df_ml, summary
