import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import xgboost as xgb

from ml.src.utils.logger import setup_logger

logger = setup_logger("WalkForwardV2")


class WalkForwardEvaluatorV2:
    """
    AtmosIQ Dataset v2 Walk-Forward Temporal Evaluator.
    Evaluates models across expanding time-series training windows (2020-2021 -> 2022, 2020-2022 -> 2023, 2020-2023 -> 2024).
    """

    def __init__(self, modeling_dir: str = "ml/data/modeling/v2", exp_dir: str = "ml/experiments/phase3e"):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.frozen_file = self.modeling_dir / "feature_dataset_frozen.csv"

    def load_dataset(self) -> pd.DataFrame:
        """Loads Dataset v2 frozen dataset."""
        assert self.frozen_file.exists(), f"Frozen dataset v2 missing: {self.frozen_file}"
        df = pd.read_csv(self.frozen_file)
        df["date_dt"] = pd.to_datetime(df["date"])
        return df.sort_values("date_dt").reset_index(drop=True)

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

    def run_walk_forward(self, feature_cols: list[str]) -> pd.DataFrame:
        """Executes 3-fold expanding window walk-forward evaluation."""
        logger.info("Executing 3-fold walk-forward temporal evaluation on Dataset v2...")
        df = self.load_dataset()

        folds = [
            {"fold": 1, "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31", "test_year": 2022},
            {"fold": 2, "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31", "test_year": 2023},
            {"fold": 3, "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "test_year": 2024}
        ]

        results_rows = []

        for f_info in folds:
            f_idx = f_info["fold"]
            tr_sub = df[df["date_dt"] <= f_info["train_end"]].copy()
            te_sub = df[(df["date_dt"] >= f_info["test_start"]) & (df["date_dt"] <= f_info["test_end"])].copy()

            X_tr = tr_sub[feature_cols]
            y_tr = tr_sub["pm25"]

            X_te = te_sub[feature_cols]
            y_te = te_sub["pm25"]

            logger.info(f"Fold {f_idx}: Train through {f_info['train_end']} ({len(X_tr)} rows) -> Predict {f_info['test_year']} ({len(X_te)} rows)...")

            # 1. Persistence Baseline: y_hat(t) = PM2.5(t-1)
            # Boundary handling: use last day of training set for day 1 of test set
            last_train_pm25 = y_tr.iloc[-1]
            pers_preds = pd.Series(np.vstack([last_train_pm25, y_te.iloc[:-1].values.reshape(-1, 1)]).ravel())
            m_pers = self.calculate_metrics(y_te, pers_preds)
            results_rows.append({"Fold": f_idx, "Test_Year": f_info["test_year"], "Train_Rows": len(X_tr), "Model": "Persistence", **m_pers})

            # 2. Ridge Regression
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_tr, y_tr)
            p_ridge = ridge.predict(X_te)
            m_ridge = self.calculate_metrics(y_te, p_ridge)
            results_rows.append({"Fold": f_idx, "Test_Year": f_info["test_year"], "Train_Rows": len(X_tr), "Model": "Ridge", **m_ridge})

            # 3. Random Forest
            rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            p_rf = rf.predict(X_te)
            m_rf = self.calculate_metrics(y_te, p_rf)
            results_rows.append({"Fold": f_idx, "Test_Year": f_info["test_year"], "Train_Rows": len(X_tr), "Model": "Random Forest", **m_rf})

            # 4. XGBoost
            xgbr = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.05, max_depth=3, reg_alpha=5.0, reg_lambda=5.0, random_state=42, n_jobs=-1)
            xgbr.fit(X_tr, y_tr)
            p_xgb = xgbr.predict(X_te)
            m_xgb = self.calculate_metrics(y_te, p_xgb)
            results_rows.append({"Fold": f_idx, "Test_Year": f_info["test_year"], "Train_Rows": len(X_tr), "Model": "XGBoost", **m_xgb})

        res_df = pd.DataFrame(results_rows)

        # Average performance across folds
        avg_df = res_df.groupby("Model").agg(
            Avg_MAE=("MAE", "mean"),
            Avg_RMSE=("RMSE", "mean"),
            Avg_R2=("R2", "mean"),
            Avg_Median_AE=("Median_AE", "mean")
        ).reset_index().round(4)
        avg_df["Fold"] = "Average (3-Fold)"
        avg_df["Test_Year"] = "2022-2024"
        avg_df["Train_Rows"] = "Expanding"

        final_res_df = pd.concat([res_df, avg_df], ignore_index=True)
        out_file = self.exp_dir / "walk_forward_results.csv"
        final_res_df.to_csv(out_file, index=False)
        logger.info(f"Walk-forward evaluation results exported to: {out_file}")

        return final_res_df


if __name__ == "__main__":
    reg_file = Path("ml/experiments/phase3c/feature_set_registry.json")
    with open(reg_file, "r") as f:
        reg_data = json.load(f)
    fset_cols = reg_data["set_b_pm25_history"]["features"]

    wf = WalkForwardEvaluatorV2()
    wf.run_walk_forward(fset_cols)
