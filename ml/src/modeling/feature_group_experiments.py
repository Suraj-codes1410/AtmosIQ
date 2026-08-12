import sys
import json
import datetime
from pathlib import Path

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureGroupExperimentsPhase3C")


class FeatureGroupExperimentEngine:
    """
    AtmosIQ Phase 3C: Feature Group Incremental Information & Model Evaluation Pipeline.
    Evaluates Random Forest and XGBoost regressors across candidate feature sets
    and measures generalization gaps and incremental value over persistence baselines.
    """

    def __init__(
        self,
        modeling_dir: str = "ml/data/modeling/v1",
        exp_dir: str = "ml/experiments/phase3c"
    ):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.pred_dir = self.exp_dir / "predictions"
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = "pm25"
        self.date_col = "date"

        # Persistence Baseline Benchmarks from Phase 3B-1
        self.persistence_val_mae = 31.9925
        self.persistence_val_rmse = 42.2714
        self.persistence_val_r2 = 0.6759

        self.persistence_test_mae = 33.5436
        self.persistence_test_rmse = 49.8988
        self.persistence_test_r2 = 0.7894

    def load_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads train, validation, and test datasets."""
        tr_df = pd.read_csv(self.modeling_dir / "train.csv")
        val_df = pd.read_csv(self.modeling_dir / "validation.csv")
        te_df = pd.read_csv(self.modeling_dir / "test.csv")
        return tr_df, val_df, te_df

    def load_feature_sets(self) -> dict[str, list[str]]:
        """Loads candidate feature sets from feature_set_registry.json."""
        reg_file = self.exp_dir / "feature_set_registry.json"
        assert reg_file.exists(), f"Feature set registry missing: {reg_file}"

        with open(reg_file, "r") as f:
            registry_data = json.load(f)

        return {k: v["features"] for k, v in registry_data.items()}

    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        """Calculates MAE, RMSE, R2, and Median Absolute Error."""
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

    def run_experiments(
        self,
        tr_df: pd.DataFrame,
        val_df: pd.DataFrame,
        te_df: pd.DataFrame,
        feature_sets: dict[str, list[str]]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs Random Forest and XGBoost evaluations across all feature sets."""
        logger.info("Executing incremental feature set experiments for Random Forest & XGBoost...")

        y_train = tr_df[self.target_col]
        y_val = val_df[self.target_col]
        y_test = te_df[self.target_col]

        dates_val = val_df[self.date_col]
        dates_test = te_df[self.date_col]

        metrics_rows = []
        comp_rows = []
        pers_comp_rows = []

        best_val_mae = float("inf")
        best_model_info = None

        for set_name, f_cols in feature_sets.items():
            f_count = len(f_cols)
            logger.info(f"Evaluating Feature Set: '{set_name}' ({f_count} features)...")

            X_tr = tr_df[f_cols]
            X_val = val_df[f_cols]
            X_te = te_df[f_cols]

            # 1. Random Forest Regressor
            rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_train)

            rf_p_tr = rf.predict(X_tr)
            rf_p_val = rf.predict(X_val)
            rf_p_te = rf.predict(X_te)

            rf_m_tr = self.calculate_metrics(y_train, rf_p_tr)
            rf_m_val = self.calculate_metrics(y_val, rf_p_val)
            rf_m_te = self.calculate_metrics(y_test, rf_p_te)

            # 2. XGBoost Regressor
            xgbr = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgbr.fit(X_tr, y_train)

            xgb_p_tr = xgbr.predict(X_tr)
            xgb_p_val = xgbr.predict(X_val)
            xgb_p_te = xgbr.predict(X_te)

            xgb_m_tr = self.calculate_metrics(y_train, xgb_p_tr)
            xgb_m_val = self.calculate_metrics(y_val, xgb_p_val)
            xgb_m_te = self.calculate_metrics(y_test, xgb_p_te)

            # Record model outputs
            for m_label, m_tr, m_val, m_te, p_val, p_te in [
                ("Random Forest", rf_m_tr, rf_m_val, rf_m_te, rf_p_val, rf_p_te),
                ("XGBoost", xgb_m_tr, xgb_m_val, xgb_m_te, xgb_p_val, xgb_p_te)
            ]:
                # Metrics Rows
                for s_name, m_dict in [("Train", m_tr), ("Validation", m_val), ("Test", m_te)]:
                    metrics_rows.append({
                        "Feature_Set": set_name,
                        "Feature_Count": f_count,
                        "Model": m_label,
                        "Split": s_name,
                        "MAE": m_dict["MAE"],
                        "RMSE": m_dict["RMSE"],
                        "R2": m_dict["R2"],
                        "Median_AE": m_dict["Median_AE"]
                    })

                # Comparison Table Rows
                tr_val_gap = round(m_tr["R2"] - m_val["R2"], 4)
                tr_te_gap = round(m_tr["R2"] - m_te["R2"], 4)

                comp_rows.append({
                    "Feature_Set": set_name,
                    "Feature_Count": f_count,
                    "Model": m_label,
                    "Train_R2": m_tr["R2"],
                    "Val_MAE": m_val["MAE"],
                    "Val_RMSE": m_val["RMSE"],
                    "Val_R2": m_val["R2"],
                    "Test_MAE": m_te["MAE"],
                    "Test_RMSE": m_te["RMSE"],
                    "Test_R2": m_te["R2"],
                    "Train_Val_R2_Gap": tr_val_gap,
                    "Train_Test_R2_Gap": tr_te_gap
                })

                # Persistence Comparison Rows
                val_mae_diff = round(self.persistence_val_mae - m_val["MAE"], 4)
                val_rmse_diff = round(self.persistence_val_rmse - m_val["RMSE"], 4)
                val_r2_diff = round(m_val["R2"] - self.persistence_val_r2, 4)

                test_mae_diff = round(self.persistence_test_mae - m_te["MAE"], 4)
                test_rmse_diff = round(self.persistence_test_rmse - m_te["RMSE"], 4)
                test_r2_diff = round(m_te["R2"] - self.persistence_test_r2, 4)

                pers_comp_rows.append({
                    "Feature_Set": set_name,
                    "Feature_Count": f_count,
                    "Model": m_label,
                    "Val_MAE": m_val["MAE"],
                    "Val_MAE_Improvement_vs_Pers": val_mae_diff,
                    "Val_RMSE": m_val["RMSE"],
                    "Val_RMSE_Improvement_vs_Pers": val_rmse_diff,
                    "Val_R2": m_val["R2"],
                    "Val_R2_Diff_vs_Pers": val_r2_diff,
                    "Test_MAE": m_te["MAE"],
                    "Test_MAE_Improvement_vs_Pers": test_mae_diff,
                    "Test_RMSE": m_te["RMSE"],
                    "Test_RMSE_Improvement_vs_Pers": test_rmse_diff,
                    "Test_R2": m_te["R2"],
                    "Test_R2_Diff_vs_Pers": test_r2_diff
                })

                # Save top prediction CSV if best validation MAE
                if m_val["MAE"] < best_val_mae:
                    best_val_mae = m_val["MAE"]
                    best_model_info = {
                        "feature_set": set_name,
                        "model_name": m_label,
                        "val_mae": m_val["MAE"],
                        "val_r2": m_val["R2"],
                        "test_mae": m_te["MAE"],
                        "test_r2": m_te["R2"],
                        "p_val": p_val,
                        "p_te": p_te
                    }

        # Export CSVs
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(self.exp_dir / "model_metrics.csv", index=False)

        comp_df = pd.DataFrame(comp_rows).sort_values(by="Val_MAE", ascending=True).reset_index(drop=True)
        comp_df.to_csv(self.exp_dir / "model_comparison.csv", index=False)

        pers_df = pd.DataFrame(pers_comp_rows).sort_values(by="Val_MAE", ascending=True).reset_index(drop=True)
        pers_df.to_csv(self.exp_dir / "persistence_comparison.csv", index=False)

        # Export predictions for best model
        if best_model_info:
            logger.info(f"Best Reduced Model on Validation: {best_model_info['model_name']} on '{best_model_info['feature_set']}' (Val MAE: {best_model_info['val_mae']:.4f})")

            b_val_df = pd.DataFrame({
                "date": dates_val,
                "actual_pm25": y_val,
                "predicted_pm25": best_model_info["p_val"],
                "residual": y_val - best_model_info["p_val"]
            })
            b_val_df.to_csv(self.pred_dir / "best_reduced_validation.csv", index=False)

            b_te_df = pd.DataFrame({
                "date": dates_test,
                "actual_pm25": y_test,
                "predicted_pm25": best_model_info["p_te"],
                "residual": y_test - best_model_info["p_te"]
            })
            b_te_df.to_csv(self.pred_dir / "best_reduced_test.csv", index=False)

        return metrics_df, comp_df, pers_df

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Executes full Phase 3C feature group experiment pipeline."""
        logger.info("=== Starting Phase 3C Feature Group Experiments ===")

        tr_df, val_df, te_df = self.load_splits()
        feature_sets = self.load_feature_sets()

        metrics_df, comp_df, pers_df = self.run_experiments(tr_df, val_df, te_df, feature_sets)

        logger.info("=== Phase 3C Feature Group Experiments Completed Successfully ===")
        return metrics_df, comp_df, pers_df


if __name__ == "__main__":
    runner = FeatureGroupExperimentEngine()
    runner.run()
