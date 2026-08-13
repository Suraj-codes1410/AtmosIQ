import sys
import json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import xgboost as xgb

from ml.src.utils.logger import setup_logger

logger = setup_logger("TunedEvaluatorPhase3G")


class TunedEvaluatorPhase3G:
    """
    AtmosIQ Phase 3G Tuned Model Evaluator & Selection Engine.
    Executes Mode A (Development Walk-Forward Folds 1 & 2) and Mode B (Final Evaluation on Locked 2024 Test Set).
    """

    def __init__(self, modeling_dir: str = "ml/data/modeling/v2", exp_dir: str = "ml/experiments/phase3g"):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.metrics_dir = self.exp_dir / "metrics"
        self.preds_dir = self.exp_dir / "predictions"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)

        self.frozen_file = self.modeling_dir / "feature_dataset_frozen.csv"
        assert self.frozen_file.exists(), f"Frozen dataset missing: {self.frozen_file}"
        self.df = pd.read_csv(self.frozen_file)
        self.df["date_dt"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date_dt").reset_index(drop=True)

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

    def instantiate_model(self, m_type: str, params: dict):
        """Instantiates a model given model type and tuned hyperparameter dictionary."""
        if m_type == "ridge":
            return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**params, random_state=42))])
        elif m_type == "elasticnet":
            return Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(**params, max_iter=10000, random_state=42))])
        elif m_type == "random_forest":
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif m_type == "xgboost":
            return xgb.XGBRegressor(objective="reg:squarederror", **params, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {m_type}")

    def evaluate_all_tuned_models(self, feature_sets: dict[str, list[str]], best_params_map: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Evaluates tuned models on Development Folds 1 & 2 (Mode A) and final 2024 Test Set (Mode B)."""
        logger.info("Evaluating tuned models on Development Walk-Forward Folds (Mode A)...")

        fold_records = []
        model_summary_records = []

        # 1. Baseline Persistence Model
        for fold, (tr_end, val_st, val_end, val_yr) in enumerate([
            ("2021-12-31", "2022-01-01", "2022-12-31", 2022),
            ("2022-12-31", "2023-01-01", "2023-12-31", 2023)
        ], start=1):
            tr_sub = self.df[self.df["date_dt"] <= tr_end]
            val_sub = self.df[(self.df["date_dt"] >= val_st) & (self.df["date_dt"] <= val_end)]

            y_tr = tr_sub["pm25"]
            y_val = val_sub["pm25"]

            p_val_pers = pd.Series(np.vstack([y_tr.iloc[-1], y_val.iloc[:-1].values.reshape(-1, 1)]).ravel())
            m_pers = self.calculate_metrics(y_val, p_val_pers)

            fold_records.append({
                "Study_Name": "persistence__pm25_lag_1d",
                "Model": "Persistence",
                "Feature_Set": "pm25_lag_1d",
                "Feature_Count": 1,
                "Fold": fold,
                "Val_Year": val_yr,
                "MAE": m_pers["MAE"],
                "RMSE": m_pers["RMSE"],
                "R2": m_pers["R2"],
                "Median_AE": m_pers["Median_AE"]
            })

        # 2. Tuned Models across Candidate Feature Sets
        for study_name, s_info in best_params_map.items():
            m_type = s_info["model_type"]
            fset_name = s_info["feature_set"]
            f_cols = feature_sets[fset_name]
            params = s_info["params"]

            m_label = " ".join(m_type.split("_")).title()

            dev_maes, dev_rmses, dev_r2s = [], [], []

            for fold, (tr_end, val_st, val_end, val_yr) in enumerate([
                ("2021-12-31", "2022-01-01", "2022-12-31", 2022),
                ("2022-12-31", "2023-01-01", "2023-12-31", 2023)
            ], start=1):
                tr_sub = self.df[self.df["date_dt"] <= tr_end]
                val_sub = self.df[(self.df["date_dt"] >= val_st) & (self.df["date_dt"] <= val_end)]

                X_tr, y_tr = tr_sub[f_cols], tr_sub["pm25"]
                X_val, y_val = val_sub[f_cols], val_sub["pm25"]

                model = self.instantiate_model(m_type, params)
                model.fit(X_tr, y_tr)

                p_tr = pd.Series(model.predict(X_tr))
                p_val = pd.Series(model.predict(X_val))

                m_val = self.calculate_metrics(y_val, p_val)
                dev_maes.append(m_val["MAE"])
                dev_rmses.append(m_val["RMSE"])
                dev_r2s.append(m_val["R2"])

                fold_records.append({
                    "Study_Name": study_name,
                    "Model": m_label,
                    "Feature_Set": fset_name,
                    "Feature_Count": len(f_cols),
                    "Fold": fold,
                    "Val_Year": val_yr,
                    "MAE": m_val["MAE"],
                    "RMSE": m_val["RMSE"],
                    "R2": m_val["R2"],
                    "Median_AE": m_val["Median_AE"]
                })

                # Save prediction CSV
                pred_df = pd.DataFrame({"date": val_sub["date"], "actual_pm25": y_val.values, "predicted_pm25": p_val.values, "residual": y_val.values - p_val.values})
                pred_df.to_csv(self.preds_dir / f"{study_name}_fold{fold}.csv", index=False)

            model_summary_records.append({
                "Study_Name": study_name,
                "Model": m_label,
                "Feature_Set": fset_name,
                "Feature_Count": len(f_cols),
                "Dev_Mean_MAE": round(float(np.mean(dev_maes)), 4),
                "Dev_MAE_Std": round(float(np.std(dev_maes)), 4),
                "Dev_Mean_RMSE": round(float(np.mean(dev_rmses)), 4),
                "Dev_Mean_R2": round(float(np.mean(dev_r2s)), 4)
            })

        fold_df = pd.DataFrame(fold_records)
        fold_df.to_csv(self.metrics_dir / "fold_metrics.csv", index=False)

        model_comp_df = pd.DataFrame(model_summary_records).sort_values("Dev_Mean_MAE").reset_index(drop=True)
        model_comp_df.to_csv(self.metrics_dir / "model_comparison.csv", index=False)
        model_comp_df.to_csv(self.metrics_dir / "feature_set_comparison.csv", index=False)
        model_comp_df.to_csv(self.metrics_dir / "stability_metrics.csv", index=False)

        # 3. SELECT BEST PRODUCTION MODEL CANDIDATE STRICTLY ON DEVELOPMENT MAE
        best_candidate = model_comp_df.iloc[0].to_dict()
        best_study_name = best_candidate["Study_Name"]
        best_info = best_params_map[best_study_name]

        logger.info(f"FINAL BEST PRODUCTION MODEL SELECTED ON DEV VALIDATION: {best_candidate['Model']} on '{best_candidate['Feature_Set']}' (Dev Mean MAE: {best_candidate['Dev_Mean_MAE']})")

        # 4. MODE B: MODE EVALUATION ON UNTOUCHED HELD-OUT 2024 TEST SET
        logger.info("Executing Mode B: ONE Final Evaluation on Locked 2024 Test Set...")

        tr_full = self.df[self.df["date_dt"] <= "2023-12-31"]
        te_2024 = self.df[(self.df["date_dt"] >= "2024-01-01") & (self.df["date_dt"] <= "2024-12-31")]

        f_cols_final = feature_sets[best_info["feature_set"]]
        X_tr_full, y_tr_full = tr_full[f_cols_final], tr_full["pm25"]
        X_te_2024, y_te_2024 = te_2024[f_cols_final], te_2024["pm25"]

        final_model = self.instantiate_model(best_info["model_type"], best_info["params"])
        final_model.fit(X_tr_full, y_tr_full)

        p_te_2024 = pd.Series(final_model.predict(X_te_2024))
        m_te_2024 = self.calculate_metrics(y_te_2024, p_te_2024)

        # Naive Persistence Baseline on 2024 Test Set
        p_pers_2024 = pd.Series(np.vstack([y_tr_full.iloc[-1], y_te_2024.iloc[:-1].values.reshape(-1, 1)]).ravel())
        m_pers_2024 = self.calculate_metrics(y_te_2024, p_pers_2024)

        pct_impr_vs_pers = round(((m_pers_2024["MAE"] - m_te_2024["MAE"]) / m_pers_2024["MAE"]) * 100, 2)

        final_test_records = [
            {
                "Model": "Persistence Baseline",
                "Feature_Set": "pm25_lag_1d",
                "Feature_Count": 1,
                "Test_MAE": m_pers_2024["MAE"],
                "Test_RMSE": m_pers_2024["RMSE"],
                "Test_R2": m_pers_2024["R2"],
                "Test_Median_AE": m_pers_2024["Median_AE"],
                "Improvement_vs_Persistence_Pct": 0.0
            },
            {
                "Model": best_candidate["Model"],
                "Feature_Set": best_candidate["Feature_Set"],
                "Feature_Count": len(f_cols_final),
                "Test_MAE": m_te_2024["MAE"],
                "Test_RMSE": m_te_2024["RMSE"],
                "Test_R2": m_te_2024["R2"],
                "Test_Median_AE": m_te_2024["Median_AE"],
                "Improvement_vs_Persistence_Pct": pct_impr_vs_pers
            }
        ]

        final_test_df = pd.DataFrame(final_test_records)
        final_test_df.to_csv(self.metrics_dir / "final_test_metrics.csv", index=False)

        # Export final predictions CSV
        final_pred_df = pd.DataFrame({
            "date": te_2024["date"],
            "actual_pm25": y_te_2024.values,
            "predicted_pm25": p_te_2024.values,
            "residual": y_te_2024.values - p_te_2024.values
        })
        final_pred_df.to_csv(self.preds_dir / "final_test_predictions.csv", index=False)

        logger.info(f"Mode B Final 2024 Test Evaluation complete: Test MAE: {m_te_2024['MAE']} (R2: {m_te_2024['R2']}) vs Persistence MAE: {m_pers_2024['MAE']} ({pct_impr_vs_pers}% improvement).")

        return fold_df, model_comp_df, {
            "best_candidate": best_candidate,
            "best_info": best_info,
            "final_model": final_model,
            "test_metrics": m_te_2024,
            "persistence_test_metrics": m_pers_2024,
            "pct_improvement": pct_impr_vs_pers,
            "feature_cols": f_cols_final
        }


if __name__ == "__main__":
    from ml.src.modeling.phase3g.feature_sets import FeatureSetManagerPhase3G
    fsets = FeatureSetManagerPhase3G().get_phase3g_feature_sets()
    with open("ml/experiments/phase3g/optuna/best_params.json") as f:
        best_p = json.load(f)

    evaluator = TunedEvaluatorPhase3G()
    evaluator.evaluate_all_tuned_models(fsets, best_p)
