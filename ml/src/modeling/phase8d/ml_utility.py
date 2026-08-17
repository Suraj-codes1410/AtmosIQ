"""
AtmosIQ Phase 8D: Downstream ML Utility Calibration Evaluator.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


class Phase8DMLUtilityEvaluator:
    """Evaluates downstream ML utility across candidate calibration subsets."""

    def __init__(self, feature_registry: List[str], random_seed: int = 42):
        self.feature_registry = list(feature_registry)
        self.random_seed = random_seed

    def evaluate_candidate_utility(
        self,
        df_real_train: pd.DataFrame,
        df_calibrated_corpus: pd.DataFrame,
        df_real_test: pd.DataFrame,
        candidate_id: str
    ) -> Dict[str, Any]:
        common = [f for f in self.feature_registry if f in df_real_train.columns and f in df_calibrated_corpus.columns and f in df_real_test.columns]

        X_real_train = df_real_train[common].values
        y_real_train = df_real_train["pm25"].values

        X_synth = df_calibrated_corpus[common].values
        y_synth = df_calibrated_corpus["pm25"].values

        X_test = df_real_test[common].values
        y_test = df_real_test["pm25"].values

        # Evaluate at primary 25% augmentation ratio
        aug_count = int(len(X_synth) * 0.25) if len(X_synth) > 0 else 0
        if aug_count > 0:
            X_tr = np.vstack([X_real_train, X_synth[:aug_count]])
            y_tr = np.concatenate([y_real_train, y_synth[:aug_count]])
        else:
            X_tr, y_tr = X_real_train, y_real_train

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

        ext_mask = (y_test >= 250.0)
        ext_mae = float(mean_absolute_error(y_test[ext_mask], preds[ext_mask])) if ext_mask.any() else 0.0

        return {
            "candidate_id": candidate_id,
            "training_samples": len(X_tr),
            "synthetic_samples_used": aug_count,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "pearson_r": pr_corr,
            "extreme_250_mae": ext_mae,
        }
