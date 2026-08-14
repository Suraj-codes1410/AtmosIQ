import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ModelTrainingPhase4H")


class ModelFactoryPhase4H:
    """
    Model Factory & Wrapper for Phase 4H Candidate Evaluation.
    """

    @staticmethod
    def load_control_model():
        """Loads the frozen production control model (MODEL_V2_PRODUCTION_CONTROL)."""
        ctrl_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
        if not ctrl_path.exists():
            raise FileNotFoundError(f"Frozen control model missing: {ctrl_path}")
        model = joblib.load(ctrl_path)
        logger.info(f"Loaded MODEL_V2_PRODUCTION_CONTROL from {ctrl_path}")
        return model

    @staticmethod
    def create_model(model_name: str, params: dict = None):
        """Creates model instance based on model_name and hyperparameters."""
        params = params.copy() if params else {}

        if model_name == "Ridge":
            alpha = params.get("alpha", 10.0)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=alpha, random_state=42))
            ])
        elif model_name == "ElasticNet":
            alpha = params.get("alpha", 0.5)
            l1_ratio = params.get("l1_ratio", 0.5)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=3000))
            ])
        elif model_name == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=params.get("n_estimators", 450),
                max_depth=params.get("max_depth", 6),
                min_samples_split=params.get("min_samples_split", 4),
                min_samples_leaf=params.get("min_samples_leaf", 4),
                max_features=params.get("max_features", 0.5),
                random_state=params.get("random_state", 42),
                n_jobs=-1
            )
        elif model_name == "XGBoost":
            model = XGBRegressor(
                n_estimators=params.get("n_estimators", 150),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.03),
                subsample=params.get("subsample", 0.7),
                colsample_bytree=params.get("colsample_bytree", 0.7),
                reg_alpha=params.get("reg_alpha", 2.0),
                reg_lambda=params.get("reg_lambda", 2.0),
                random_state=params.get("random_state", 42),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        return model
