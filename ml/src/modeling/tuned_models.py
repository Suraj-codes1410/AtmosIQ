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

logger = setup_logger("TunedModelsPhase3D")


class TunedModelEvaluator:
    """
    AtmosIQ Phase 3D: Tuned Model Trainer & Evaluator.
    Trains candidate models using Optuna-selected best hyperparameters and evaluates
    Train, Validation, and Test metrics with zero test leakage during selection.
    """

    def __init__(
        self,
        modeling_dir: str = "ml/data/modeling/v1",
        exp_dir: str = "ml/experiments/phase3d"
    ):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.best_params_file = self.exp_dir / "best_parameters" / "best_params.json"
        self.metrics_dir = self.exp_dir / "metrics"
        self.pred_dir = self.exp_dir / "predictions"

        self.target_col = "pm25"
        self.date_col = "date"

        # Baseline Benchmarks for Comparison
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

    def load_best_params(self) -> dict:
        """Loads best parameters from json."""
        assert self.best_params_file.exists(), f"Best params file missing: {self.best_params_file}"
        with open(self.best_params_file, "r") as f:
            best_params = json.load(f)
        return best_params

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

    def train_and_evaluate(
        self,
        feature_sets: dict[str, list[str]]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Trains all 12 tuned models on X_train and evaluates on Train, Validation, Test."""
        logger.info("Training tuned models with Optuna hyperparameters...")
        tr_df, val_df, te_df = self.load_splits()
        best_params_map = self.load_best_params()

        y_train = tr_df[self.target_col]
        y_val = val_df[self.target_col]
        y_test = te_df[self.target_col]

        dates_val = val_df[self.date_col]
        dates_test = te_df[self.date_col]

        val_rows = []
        test_rows = []
        comp_rows = []
        overfit_rows = []

        all_predictions = {}
        best_candidate = None
        lowest_val_mae = float("inf")

        for key, study_info in best_params_map.items():
            m_name = study_info["model_name"]
            fset_name = study_info["feature_set"]
            f_cols = feature_sets[fset_name]
            params = study_info["best_params"]

            X_tr = tr_df[f_cols].copy()
            X_val = val_df[f_cols].copy()
            X_te = te_df[f_cols].copy()

            # Instantiate model
            if m_name == "Ridge":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=params["alpha"], random_state=42))
                ])
            elif m_name == "ElasticNet":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=10000, random_state=42))
                ])
            elif m_name == "Random Forest":
                m_feat = params["max_features_float"] if params["max_features_type"] == "float" else "sqrt"
                model = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_features=m_feat,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            elif m_name == "XGBoost":
                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    min_child_weight=params["min_child_weight"],
                    subsample=params["subsample"],
                    colsample_bytree=params["colsample_bytree"],
                    gamma=params["gamma"],
                    reg_alpha=params["reg_alpha"],
                    reg_lambda=params["reg_lambda"],
                    random_state=42,
                    n_jobs=-1
                )

            # Fit strictly on X_train
            model.fit(X_tr, y_train)

            p_tr = model.predict(X_tr)
            p_val = model.predict(X_val)
            p_te = model.predict(X_te)

            m_tr = self.calculate_metrics(y_train, p_tr)
            m_val = self.calculate_metrics(y_val, p_val)
            m_te = self.calculate_metrics(y_test, p_te)

            # Save validation predictions CSV
            pred_slug = f"{m_name.lower().replace(' ', '_')}_{fset_name}_validation.csv"
            val_pred_df = pd.DataFrame({
                "date": dates_val,
                "actual_pm25": y_val,
                "predicted_pm25": p_val,
                "residual": y_val - p_val
            })
            val_pred_df.to_csv(self.pred_dir / pred_slug, index=False)

            val_rows.append({
                "Model": m_name,
                "Feature_Set": fset_name,
                "Feature_Count": len(f_cols),
                "MAE": m_val["MAE"],
                "RMSE": m_val["RMSE"],
                "R2": m_val["R2"],
                "Median_AE": m_val["Median_AE"]
            })

            test_rows.append({
                "Model": m_name,
                "Feature_Set": fset_name,
                "Feature_Count": len(f_cols),
                "MAE": m_te["MAE"],
                "RMSE": m_te["RMSE"],
                "R2": m_te["R2"],
                "Median_AE": m_te["Median_AE"]
            })

            val_mae_imp = round(self.persistence_val_mae - m_val["MAE"], 4)
            val_rmse_imp = round(self.persistence_val_rmse - m_val["RMSE"], 4)
            val_r2_diff = round(m_val["R2"] - self.persistence_val_r2, 4)

            comp_rows.append({
                "Model": m_name,
                "Feature_Set": fset_name,
                "Feature_Count": len(f_cols),
                "Val_MAE": m_val["MAE"],
                "Val_RMSE": m_val["RMSE"],
                "Val_R2": m_val["R2"],
                "Val_MAE_Improvement_vs_Pers": val_mae_imp,
                "Val_R2_Diff_vs_Pers": val_r2_diff,
                "Test_MAE": m_te["MAE"],
                "Test_RMSE": m_te["RMSE"],
                "Test_R2": m_te["R2"]
            })

            tr_val_gap = round(m_tr["R2"] - m_val["R2"], 4)
            tr_te_gap = round(m_tr["R2"] - m_te["R2"], 4)
            overfit_rows.append({
                "Model": m_name,
                "Feature_Set": fset_name,
                "Feature_Count": len(f_cols),
                "Train_R2": m_tr["R2"],
                "Val_R2": m_val["R2"],
                "Test_R2": m_te["R2"],
                "Train_Val_R2_Gap": tr_val_gap,
                "Train_Test_R2_Gap": tr_te_gap
            })

            # Check if lowest validation MAE (Model Selection strictly on Validation)
            if m_val["MAE"] < lowest_val_mae:
                lowest_val_mae = m_val["MAE"]
                best_candidate = {
                    "model_name": m_name,
                    "feature_set": fset_name,
                    "feature_count": len(f_cols),
                    "model": model,
                    "val_mae": m_val["MAE"],
                    "val_rmse": m_val["RMSE"],
                    "val_r2": m_val["R2"],
                    "p_val": p_val,
                    "p_te": p_te,
                    "m_te": m_te
                }

        # Export Metric CSVs
        val_df_out = pd.DataFrame(val_rows).sort_values("MAE").reset_index(drop=True)
        val_df_out.to_csv(self.metrics_dir / "validation_metrics.csv", index=False)

        test_df_out = pd.DataFrame(test_rows).sort_values("MAE").reset_index(drop=True)
        test_df_out.to_csv(self.metrics_dir / "test_metrics.csv", index=False)

        comp_df_out = pd.DataFrame(comp_rows).sort_values("Val_MAE").reset_index(drop=True)
        comp_df_out.to_csv(self.metrics_dir / "model_comparison.csv", index=False)

        overfit_df_out = pd.DataFrame(overfit_rows).sort_values("Train_Val_R2_Gap").reset_index(drop=True)
        overfit_df_out.to_csv(self.metrics_dir / "overfitting_analysis.csv", index=False)

        # Freeze Final Candidate Model & Evaluate ONCE on Test Set
        logger.info(f"Final Model Candidate Selected strictly on Validation: {best_candidate['model_name']} on '{best_candidate['feature_set']}' (Val MAE: {best_candidate['val_mae']:.4f})")

        final_test_pred_df = pd.DataFrame({
            "date": dates_test,
            "actual_pm25": y_test,
            "predicted_pm25": best_candidate["p_te"],
            "residual": y_test - best_candidate["p_te"]
        })
        final_test_slug = f"{best_candidate['model_name'].lower().replace(' ', '_')}_{best_candidate['feature_set']}_test.csv"
        final_test_pred_df.to_csv(self.pred_dir / final_test_slug, index=False)

        return comp_df_out, overfit_df_out, val_df_out, best_candidate


if __name__ == "__main__":
    from ml.src.modeling.feature_sets import FeatureSetManager
    fsets = FeatureSetManager().get_phase3d_feature_sets()

    evaluator = TunedModelEvaluator()
    comp_df, overfit_df, val_df, best_cand = evaluator.train_and_evaluate(fsets)
