"""
Phase 0 Sanity Check Test
Verifies that the Python environment and imports work properly.
"""

def test_imports():
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost
    import lightgbm
    import shap
    import optuna
    import mlflow
    import fastapi
    
    assert pd is not None
    assert np is not None
    assert sklearn is not None
    assert xgboost is not None
    assert lightgbm is not None
    assert shap is not None
    assert optuna is not None
    assert mlflow is not None
    assert fastapi is not None

def test_config_import():
    from ml.configs.config import config
    assert config.PROJECT_NAME == "atmosIQ"
