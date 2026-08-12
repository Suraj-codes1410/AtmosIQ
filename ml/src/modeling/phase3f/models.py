import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from ml.src.utils.logger import setup_logger

logger = setup_logger("ModelsPhase3F")


class ModelFactoryPhase3F:
    """
    AtmosIQ Phase 3F Model Factory.
    Instantiates conservative baseline models (Persistence, Ridge, ElasticNet, Random Forest, XGBoost)
    with fixed hyperparameters for fair comparative evaluation across feature groups.
    """

    @staticmethod
    def get_models() -> dict:
        """Returns dictionary of un-fitted model instances / pipelines."""
        return {
            "Persistence": "PERSISTENCE_BASELINE",
            "Ridge": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=42))
            ]),
            "ElasticNet": Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42))
            ]),
            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                min_child_weight=5,
                reg_alpha=5.0,
                reg_lambda=5.0,
                random_state=42,
                n_jobs=-1
            )
        }
