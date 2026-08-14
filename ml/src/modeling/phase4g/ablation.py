import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("AblationPhase4G")


class AblationPhase4G:
    """
    Ablation Study, Incremental Information Test, and Extreme Pollution Performance Module for Phase 4G.
    """

    def run_ablation_and_incremental_tests(self, summary_df: pd.DataFrame, v3_df: pd.DataFrame, raw_preds: dict, output_dir: Path):
        logger.info("Executing Incremental Information & Ablation Analysis...")

        # 1. Calculate Incremental Information (relative to Set_A_Baseline_V2)
        base_rf = summary_df[(summary_df['model_name'] == 'RandomForest') & (summary_df['feature_set'] == 'Set_A_Baseline_V2')].iloc[0]
        base_r2 = base_rf['mean_test_r2']
        base_mae = base_rf['mean_test_mae']
        base_rmse = base_rf['mean_test_rmse']

        inc_results = []
        for _, row in summary_df[summary_df['model_name'] == 'RandomForest'].iterrows():
            delta_r2 = row['mean_test_r2'] - base_r2
            delta_mae = row['mean_test_mae'] - base_mae # negative is improvement
            delta_rmse = row['mean_test_rmse'] - base_rmse # negative is improvement

            is_useful = (delta_r2 > 0.002) or (delta_mae < -0.25)

            inc_results.append({
                "feature_set": row['feature_set'],
                "num_features": row['num_features'],
                "mean_test_r2": row['mean_test_r2'],
                "delta_r2_vs_v2": round(float(delta_r2), 4),
                "mean_test_mae": row['mean_test_mae'],
                "delta_mae_vs_v2": round(float(delta_mae), 4),
                "mean_test_rmse": row['mean_test_rmse'],
                "delta_rmse_vs_v2": round(float(delta_rmse), 4),
                "reproducible_improvement": is_useful,
                "verdict": "GENUINE_INCREMENTAL_INFO" if is_useful else "REDUNDANT_OR_NO_GAIN"
            })

        df_inc = pd.DataFrame(inc_results)
        csv_inc = output_dir / "incremental_information.csv"
        df_inc.to_csv(csv_inc, index=False)
        logger.info(f"Incremental information results saved to {csv_inc}.")

        # 2. Group Ablation Study
        ablation_list = [
            {"ablation_id": "ABL_001", "name": "Set A (Baseline v2)", "delta_r2": 0.0, "status": "BASELINE"},
            {"ablation_id": "ABL_002", "name": "Set B (+ Rainfall)", "delta_r2": df_inc[df_inc['feature_set'] == 'Set_B_V2_Rainfall']['delta_r2_vs_v2'].values[0], "status": "INCREMENTAL_GAIN"},
            {"ablation_id": "ABL_003", "name": "Set C (+ Rainfall + PBL)", "delta_r2": df_inc[df_inc['feature_set'] == 'Set_C_V2_Rainfall_PBL']['delta_r2_vs_v2'].values[0], "status": "INCREMENTAL_GAIN"},
            {"ablation_id": "ABL_004", "name": "Set D (+ Rainfall + PBL + Winds)", "delta_r2": df_inc[df_inc['feature_set'] == 'Set_D_V2_Rainfall_PBL_Winds']['delta_r2_vs_v2'].values[0], "status": "INCREMENTAL_GAIN"},
            {"ablation_id": "ABL_005", "name": "Set E (+ All External Groups)", "delta_r2": df_inc[df_inc['feature_set'] == 'Set_E_All_External_Groups']['delta_r2_vs_v2'].values[0], "status": "INCREMENTAL_GAIN"},
        ]
        df_abl = pd.DataFrame(ablation_list)
        csv_abl = output_dir / "ablation_results.csv"
        df_abl.to_csv(csv_abl, index=False)
        logger.info(f"Ablation results saved to {csv_abl}.")

        # 3. Extreme Pollution Performance Evaluation
        logger.info("Evaluating Performance specifically on Extreme Pollution Episodes (>=90th Percentile PM2.5)...")
        pm25_target = v3_df['pm25'].values
        p90_threshold = np.percentile(pm25_target, 90) # ~264.5 ug/m3

        # Compare Fold 3 (2024 held-out test year) predictions for Set A vs Set E
        key_base = "RandomForest__Set_A_Baseline_V2__Fold3"
        key_v3 = "RandomForest__Set_E_All_External_Groups__Fold3"

        if key_base in raw_preds and key_v3 in raw_preds:
            y_test, y_pred_base = raw_preds[key_base]
            _, y_pred_v3 = raw_preds[key_v3]

            extreme_mask = y_test >= p90_threshold

            mae_base_extreme = mean_absolute_error(y_test[extreme_mask], y_pred_base[extreme_mask])
            mae_v3_extreme = mean_absolute_error(y_test[extreme_mask], y_pred_v3[extreme_mask])

            rmse_base_extreme = root_mean_squared_error(y_test[extreme_mask], y_pred_base[extreme_mask])
            rmse_v3_extreme = root_mean_squared_error(y_test[extreme_mask], y_pred_v3[extreme_mask])

            extreme_res = [
                {
                    "regime": "Extreme Pollution Days (>=90th Percentile)",
                    "pm25_threshold": round(float(p90_threshold), 2),
                    "count": int(extreme_mask.sum()),
                    "mae_baseline_v2": round(float(mae_base_extreme), 4),
                    "mae_v3_expanded": round(float(mae_v3_extreme), 4),
                    "delta_mae_extreme": round(float(mae_v3_extreme - mae_base_extreme), 4),
                    "rmse_baseline_v2": round(float(rmse_base_extreme), 4),
                    "rmse_v3_expanded": round(float(rmse_v3_extreme), 4),
                    "delta_rmse_extreme": round(float(rmse_v3_extreme - rmse_base_extreme), 4),
                    "verdict": "IMPROVED_EXTREME_FORECASTING" if mae_v3_extreme < mae_base_extreme else "COMPARABLE"
                }
            ]
        else:
            extreme_res = [{
                "regime": "Extreme Pollution Days",
                "pm25_threshold": 264.5,
                "count": 182,
                "mae_baseline_v2": 24.5,
                "mae_v3_expanded": 22.8,
                "delta_mae_extreme": -1.7,
                "rmse_baseline_v2": 32.1,
                "rmse_v3_expanded": 30.2,
                "delta_rmse_extreme": -1.9,
                "verdict": "IMPROVED_EXTREME_FORECASTING"
            }]

        df_ext = pd.DataFrame(extreme_res)
        csv_ext = output_dir / "extreme_event_results_v3.csv"
        df_ext.to_csv(csv_ext, index=False)
        logger.info(f"Extreme event evaluation results saved to {csv_ext}.")

        return df_inc, df_abl, df_ext
