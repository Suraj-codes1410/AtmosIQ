import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4h.model_training import ModelFactoryPhase4H

logger = setup_logger("WalkForwardPhase4H")


class WalkForwardPhase4H:
    """
    Chronological Walk-Forward Evaluation Engine for Phase 4H.
    Evaluates control and candidate models across 3 expanding temporal folds:
    - Fold 1: Train 2020-2021, Test 2022
    - Fold 2: Train 2020-2022, Test 2023
    - Fold 3: Train 2020-2023, Test 2024
    """

    def __init__(self, df_v2: pd.DataFrame, df_v3: pd.DataFrame):
        self.df_v2 = df_v2.copy()
        self.df_v2['date'] = pd.to_datetime(self.df_v2['date'])
        self.df_v2 = self.df_v2.sort_values('date').reset_index(drop=True)

        self.df_v3 = df_v3.copy()
        self.df_v3['date'] = pd.to_datetime(self.df_v3['date'])
        self.df_v3 = self.df_v3.sort_values('date').reset_index(drop=True)

        self.folds = [
            {
                "fold": 1,
                "train_years": "2020-2021",
                "test_year": 2022,
                "train_mask_v2": self.df_v2['date'].dt.year.isin([2020, 2021]),
                "test_mask_v2": self.df_v2['date'].dt.year == 2022,
                "train_mask_v3": self.df_v3['date'].dt.year.isin([2020, 2021]),
                "test_mask_v3": self.df_v3['date'].dt.year == 2022
            },
            {
                "fold": 2,
                "train_years": "2020-2022",
                "test_year": 2023,
                "train_mask_v2": self.df_v2['date'].dt.year.isin([2020, 2021, 2022]),
                "test_mask_v2": self.df_v2['date'].dt.year == 2023,
                "train_mask_v3": self.df_v3['date'].dt.year.isin([2020, 2021, 2022]),
                "test_mask_v3": self.df_v3['date'].dt.year == 2023
            },
            {
                "fold": 3,
                "train_years": "2020-2023",
                "test_year": 2024,
                "train_mask_v2": self.df_v2['date'].dt.year.isin([2020, 2021, 2022, 2023]),
                "test_mask_v2": self.df_v2['date'].dt.year == 2024,
                "train_mask_v3": self.df_v3['date'].dt.year.isin([2020, 2021, 2022, 2023]),
                "test_mask_v3": self.df_v3['date'].dt.year == 2024
            }
        ]

    def evaluate_control_model(self, v2_features: list) -> tuple:
        """Evaluates frozen MODEL_V2_PRODUCTION_CONTROL across all folds and test years."""
        logger.info("Evaluating MODEL_V2_PRODUCTION_CONTROL on Dataset v2...")
        control_model = ModelFactoryPhase4H.load_control_model()

        fold_results = []
        out_of_sample_preds = {}

        for fold in self.folds:
            test_df = self.df_v2[fold["test_mask_v2"]].copy()
            X_test = test_df[v2_features].fillna(0.0)
            y_test = test_df['pm25'].values
            dates = test_df['date'].values

            preds = control_model.predict(X_test)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(root_mean_squared_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            medae = float(median_absolute_error(y_test, preds))

            res = {
                "fold": fold["fold"],
                "train_years": fold["train_years"],
                "test_year": fold["test_year"],
                "model_name": "Frozen_RF_v2",
                "dataset_version": "v2",
                "feature_set": "Candidate_A_V2_Baseline",
                "num_features": len(v2_features),
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
                "test_medae": medae,
                "generalization_gap": 0.0
            }
            fold_results.append(res)
            out_of_sample_preds[fold["test_year"]] = pd.DataFrame({
                "date": dates,
                "y_true": y_test,
                "y_pred": preds
            })

            logger.info(f"Control v2 Year {fold['test_year']}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MedAE={medae:.4f}")

        # Regression check 2024 test MAE
        test_2024_mae = fold_results[2]["test_mae"]
        expected_2024_mae = 26.7655
        if abs(test_2024_mae - expected_2024_mae) > 0.01:
            logger.warning(f"REGRESSION DISCREPANCY DETECTED: 2024 Test MAE = {test_2024_mae:.4f}, expected {expected_2024_mae}")
        else:
            logger.info(f"REGRESSION CHECK PASSED: 2024 Test MAE = {test_2024_mae:.4f} (matches {expected_2024_mae})")

        return fold_results, out_of_sample_preds

    def evaluate_candidate_model(self, model_name: str, feature_set_name: str, features: list, params: dict = None) -> tuple:
        """Evaluates a candidate model trained on Dataset v3 across walk-forward folds."""
        fold_results = []
        out_of_sample_preds = {}

        for fold in self.folds:
            train_df = self.df_v3[fold["train_mask_v3"]].copy()
            test_df = self.df_v3[fold["test_mask_v3"]].copy()

            X_train = train_df[features].fillna(0.0)
            y_train = train_df['pm25'].values

            X_test = test_df[features].fillna(0.0)
            y_test = test_df['pm25'].values
            dates = test_df['date'].values

            model = ModelFactoryPhase4H.create_model(model_name, params)
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

            train_r2 = float(r2_score(y_train, train_preds))
            test_r2 = float(r2_score(y_test, test_preds))
            mae = float(mean_absolute_error(y_test, test_preds))
            rmse = float(root_mean_squared_error(y_test, test_preds))
            medae = float(median_absolute_error(y_test, test_preds))

            res = {
                "fold": fold["fold"],
                "train_years": fold["train_years"],
                "test_year": fold["test_year"],
                "model_name": model_name,
                "dataset_version": "v3",
                "feature_set": feature_set_name,
                "num_features": len(features),
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_mae": mae,
                "test_rmse": rmse,
                "test_medae": medae,
                "generalization_gap": train_r2 - test_r2,
                "fitted_model": model
            }
            fold_results.append(res)
            out_of_sample_preds[fold["test_year"]] = pd.DataFrame({
                "date": dates,
                "y_true": y_test,
                "y_pred": test_preds
            })

        return fold_results, out_of_sample_preds
