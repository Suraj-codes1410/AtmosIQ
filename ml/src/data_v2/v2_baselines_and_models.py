import sys
import json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import xgboost as xgb

from ml.src.utils.logger import setup_logger

logger = setup_logger("V2ModelsPhase3E")


class DatasetV2ModelEvaluator:
    """
    AtmosIQ Phase 3E: Dataset v2 Baseline & Incremental Feature Group Evaluator.
    Evaluates Persistence, Ridge, ElasticNet, Random Forest, and XGBoost across 5-year Dataset v2.
    """

    def __init__(self, modeling_dir: str = "ml/data/modeling/v2", exp_dir: str = "ml/experiments/phase3e"):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = "pm25"
        self.date_col = "date"

    def load_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads Train (1,096), Validation (365), and Test (366) splits."""
        tr_df = pd.read_csv(self.modeling_dir / "train.csv")
        val_df = pd.read_csv(self.modeling_dir / "validation.csv")
        te_df = pd.read_csv(self.modeling_dir / "test.csv")
        return tr_df, val_df, te_df

    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        """Calculates MAE, RMSE, R2, and Median AE."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        med_ae = float(median_absolute_error(y_true, y_pred))

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Median_AE": round(med_ae, 4)
        }

    def run_v2_model_evaluations(self, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs baseline models and incremental feature group experiments on Dataset v2."""
        logger.info("Evaluating baseline & ML models across Dataset v2 temporal partitions...")
        tr_df, val_df, te_df = self.load_splits()

        y_tr = tr_df[self.target_col]
        y_val = val_df[self.target_col]
        y_te = te_df[self.target_col]

        metrics_rows = []
        fg_exp_rows = []

        # 1. Persistence Baseline: y_hat(t) = PM2.5(t-1)
        last_tr_actual = y_tr.iloc[-1]
        p_val_pers = pd.Series(np.vstack([last_tr_actual, y_val.iloc[:-1].values.reshape(-1, 1)]).ravel())
        m_val_pers = self.calculate_metrics(y_val, p_val_pers)

        last_val_actual = y_val.iloc[-1]
        p_te_pers = pd.Series(np.vstack([last_val_actual, y_te.iloc[:-1].values.reshape(-1, 1)]).ravel())
        m_te_pers = self.calculate_metrics(y_te, p_te_pers)

        for s_name, m_dict in [("Validation", m_val_pers), ("Test", m_te_pers)]:
            metrics_rows.append({
                "Model": "Persistence",
                "Feature_Set": "pm25_lag_1d",
                "Feature_Count": 1,
                "Split": s_name,
                "MAE": m_dict["MAE"],
                "RMSE": m_dict["RMSE"],
                "R2": m_dict["R2"],
                "Median_AE": m_dict["Median_AE"]
            })

        # 2. Machine Learning Models across Candidate Feature Sets
        for fset_name, f_cols in feature_sets.items():
            f_cnt = len(f_cols)
            X_tr = tr_df[f_cols]
            X_val = val_df[f_cols]
            X_te = te_df[f_cols]

            # A. Ridge Regression
            ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0, random_state=42))])
            ridge.fit(X_tr, y_tr)
            p_ridge_val = ridge.predict(X_val)
            p_ridge_te = ridge.predict(X_te)
            m_r_val = self.calculate_metrics(y_val, p_ridge_val)
            m_r_te = self.calculate_metrics(y_te, p_ridge_te)

            # B. ElasticNet
            elastic = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42))])
            elastic.fit(X_tr, y_tr)
            p_el_val = elastic.predict(X_val)
            p_el_te = elastic.predict(X_te)
            m_el_val = self.calculate_metrics(y_val, p_el_val)
            m_el_te = self.calculate_metrics(y_te, p_el_te)

            # C. Random Forest
            rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            p_rf_val = rf.predict(X_val)
            p_rf_te = rf.predict(X_te)
            m_rf_val = self.calculate_metrics(y_val, p_rf_val)
            m_rf_te = self.calculate_metrics(y_te, p_rf_te)

            # D. XGBoost
            xgbr = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=5, reg_alpha=5.0, reg_lambda=5.0, random_state=42, n_jobs=-1)
            xgbr.fit(X_tr, y_tr)
            p_xgb_val = xgbr.predict(X_val)
            p_xgb_te = xgbr.predict(X_te)
            m_xgb_val = self.calculate_metrics(y_val, p_xgb_val)
            m_xgb_te = self.calculate_metrics(y_te, p_xgb_te)

            for m_label, m_val, m_te in [
                ("Ridge", m_r_val, m_r_te),
                ("ElasticNet", m_el_val, m_el_te),
                ("Random Forest", m_rf_val, m_rf_te),
                ("XGBoost", m_xgb_val, m_xgb_te)
            ]:
                metrics_rows.append({"Model": m_label, "Feature_Set": fset_name, "Feature_Count": f_cnt, "Split": "Validation", **m_val})
                metrics_rows.append({"Model": m_label, "Feature_Set": fset_name, "Feature_Count": f_cnt, "Split": "Test", **m_te})

                fg_exp_rows.append({
                    "Model": m_label,
                    "Feature_Set": fset_name,
                    "Feature_Count": f_cnt,
                    "Val_MAE": m_val["MAE"],
                    "Val_RMSE": m_val["RMSE"],
                    "Val_R2": m_val["R2"],
                    "Test_MAE": m_te["MAE"],
                    "Test_RMSE": m_te["RMSE"],
                    "Test_R2": m_te["R2"]
                })

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(self.exp_dir / "v2_model_metrics.csv", index=False)

        fg_df = pd.DataFrame(fg_exp_rows).sort_values("Val_MAE").reset_index(drop=True)
        fg_df.to_csv(self.exp_dir / "v2_feature_group_experiments.csv", index=False)

        logger.info(f"Dataset v2 model evaluations completed and saved to: {self.exp_dir}")
        return metrics_df, fg_df


if __name__ == "__main__":
    reg_file = Path("ml/experiments/phase3c/feature_set_registry.json")
    with open(reg_file, "r") as f:
        reg_data = json.load(f)

    fsets = {
        "set_a_persistence": ["pm25_lag_1d"],
        "set_b_pm25_history": reg_data["set_b_pm25_history"]["features"],
        "domain_reduced": reg_data["domain_reduced"]["features"],
        "set_f_full_safe": reg_data["set_f_full_safe"]["features"]
    }

    evaluator = DatasetV2ModelEvaluator()
    evaluator.run_v2_model_evaluations(fsets)
