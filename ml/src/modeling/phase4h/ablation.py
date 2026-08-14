import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4h.walk_forward import WalkForwardPhase4H

logger = setup_logger("AblationPhase4H")


class AblationEnginePhase4H:
    """
    External Environmental Feature Ablation Engine for Phase 4H.
    Evaluates controlled feature group additions:
    - Model A: v2 features only
    - Model B: v2 + rainfall features
    - Model C: v2 + PBL/ventilation features
    - Model D: v2 + all external environmental features
    """

    def __init__(self, df_v2: pd.DataFrame, df_v3: pd.DataFrame, base_v2_features: list):
        self.df_v2 = df_v2
        self.df_v3 = df_v3
        self.base_v2 = [f for f in base_v2_features if f in df_v3.columns]
        self.wf = WalkForwardPhase4H(df_v2, df_v3)

    def run_ablation_study(self, candidate_model_name: str = "RandomForest", best_params: dict = None) -> pd.DataFrame:
        logger.info(f"Executing External Feature Ablation Study for {candidate_model_name}...")

        rainfall_cols = ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rain_event_1d", "washout_index_3d"]
        pbl_cols = ["pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"]
        wind_aod_cols = ["wind_u_component_1d", "wind_v_component_1d", "upwind_stubble_quadrant_1d", "aod_550_1d", "aod_roll_mean_3d"]

        ablation_configs = {
            "Model_A_v2_only": self.base_v2,
            "Model_B_v2_plus_rainfall": self.base_v2 + [f for f in rainfall_cols if f in self.df_v3.columns],
            "Model_C_v2_plus_pbl": self.base_v2 + [f for f in pbl_cols if f in self.df_v3.columns],
            "Model_D_v2_plus_all_external": self.base_v2 + [f for f in (rainfall_cols + pbl_cols + wind_aod_cols) if f in self.df_v3.columns]
        }

        # Baseline metrics from Model A
        res_a, _ = self.wf.evaluate_candidate_model(candidate_model_name, "Model_A_v2_only", ablation_configs["Model_A_v2_only"], best_params)
        base_mae = np.mean([r["test_mae"] for r in res_a])
        base_rmse = np.mean([r["test_rmse"] for r in res_a])
        base_r2 = np.mean([r["test_r2"] for r in res_a])

        records = []
        for config_name, f_list in ablation_configs.items():
            fold_results, _ = self.wf.evaluate_candidate_model(candidate_model_name, config_name, f_list, best_params)

            mean_mae = float(np.mean([r["test_mae"] for r in fold_results]))
            mean_rmse = float(np.mean([r["test_rmse"] for r in fold_results]))
            mean_r2 = float(np.mean([r["test_r2"] for r in fold_results]))

            records.append({
                "ablation_config": config_name,
                "model_name": candidate_model_name,
                "num_features": len(f_list),
                "mean_mae": round(mean_mae, 4),
                "mean_rmse": round(mean_rmse, 4),
                "mean_r2": round(mean_r2, 4),
                "delta_mae_vs_v2_only": round(mean_mae - base_mae, 4),
                "delta_rmse_vs_v2_only": round(mean_rmse - base_rmse, 4),
                "delta_r2_vs_v2_only": round(mean_r2 - base_r2, 4)
            })

            logger.info(f"Ablation {config_name}: MAE={mean_mae:.4f}, R2={mean_r2:.4f}, ΔMAE={mean_mae - base_mae:.4f}")

        return pd.DataFrame(records)
