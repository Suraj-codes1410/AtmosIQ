import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("WalkForwardPhase4G")


class WalkForwardPhase4G:
    """
    Chronological Walk-Forward Evaluation Engine for Phase 4G.
    Enforces strict temporal order across 3 expanding windows (2020-2024).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        self.target_col = 'target_pm25' if 'target_pm25' in self.df.columns else 'pm25'

        # Build folds
        self.folds = [
            {
                "fold": 1,
                "train_years": "2020-2021",
                "test_year": "2022",
                "train_mask": self.df['date'].dt.year.isin([2020, 2021]),
                "test_mask": self.df['date'].dt.year == 2022
            },
            {
                "fold": 2,
                "train_years": "2020-2022",
                "test_year": "2023",
                "train_mask": self.df['date'].dt.year.isin([2020, 2021, 2022]),
                "test_mask": self.df['date'].dt.year == 2023
            },
            {
                "fold": 3,
                "train_years": "2020-2023",
                "test_year": "2024",
                "train_mask": self.df['date'].dt.year.isin([2020, 2021, 2022, 2023]),
                "test_mask": self.df['date'].dt.year == 2024
            }
        ]

    def evaluate_model_on_fold(self, model_name: str, feature_set_name: str, features: list, fold_info: dict) -> dict:
        train_df = self.df[fold_info["train_mask"]].copy()
        test_df = self.df[fold_info["test_mask"]].copy()

        X_train = train_df[features].fillna(0.0)
        y_train = train_df[self.target_col].values

        X_test = test_df[features].fillna(0.0)
        y_test = test_df[self.target_col].values

        if model_name == "Persistence":
            # Lag-1d persistence
            if "pm25_lag_1d" in train_df.columns:
                y_pred_train = train_df["pm25_lag_1d"].values
                y_pred_test = test_df["pm25_lag_1d"].values
            else:
                y_pred_train = np.roll(y_train, 1); y_pred_train[0] = y_train[0]
                y_pred_test = np.roll(y_test, 1); y_pred_test[0] = y_test[0]
            model_obj = None
        elif model_name == "Ridge":
            model_obj = Ridge(alpha=1.0, random_state=42)
            model_obj.fit(X_train, y_train)
            y_pred_train = model_obj.predict(X_train)
            y_pred_test = model_obj.predict(X_test)
        elif model_name == "ElasticNet":
            model_obj = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
            model_obj.fit(X_train, y_train)
            y_pred_train = model_obj.predict(X_train)
            y_pred_test = model_obj.predict(X_test)
        elif model_name == "RandomForest":
            model_obj = RandomForestRegressor(
                n_estimators=450,
                max_depth=9,
                min_samples_split=3,
                min_samples_leaf=3,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            )
            model_obj.fit(X_train, y_train)
            y_pred_train = model_obj.predict(X_train)
            y_pred_test = model_obj.predict(X_test)
        elif model_name == "XGBoost":
            model_obj = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            model_obj.fit(X_train, y_train)
            y_pred_train = model_obj.predict(X_train)
            y_pred_test = model_obj.predict(X_test)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        # Metrics
        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred_test))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_rmse = float(root_mean_squared_error(y_test, y_pred_test))
        test_medae = float(median_absolute_error(y_test, y_pred_test))

        gen_gap = train_r2 - test_r2

        return {
            "fold": fold_info["fold"],
            "train_years": fold_info["train_years"],
            "test_year": fold_info["test_year"],
            "model_name": model_name,
            "feature_set": feature_set_name,
            "num_features": len(features),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_medae": test_medae,
            "generalization_gap": gen_gap,
            "y_test": y_test,
            "y_pred_test": y_pred_test,
            "fitted_model": model_obj
        }
