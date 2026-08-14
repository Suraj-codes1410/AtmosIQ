import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("SHAPConsistencyPhase4D")


class SHAPConsistencyPhase4D:
    """
    AtmosIQ Phase 4D SHAP-Counterfactual Consistency Evaluator.
    Compares directional agreement between Phase 4B SHAP attributions and counterfactual model prediction changes Δŷ.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_consistency(self, group_shap_df: pd.DataFrame, counterfactual_results_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates directional consistency between observed SHAP and counterfactual Δŷ."""
        logger.info("Evaluating directional consistency between Phase 4B SHAP and counterfactual Δŷ...")

        rows = []
        for i in range(len(counterfactual_results_df)):
            cf_row = counterfactual_results_df.iloc[i]
            dt = cf_row["date"]
            grp = cf_row["target_group"]
            delta = float(cf_row["delta_prediction"])
            scenario = cf_row["scenario"]

            # Match Phase 4B SHAP value
            shap_col = f"{grp}_shap"
            match_shap = group_shap_df[group_shap_df["date"] == dt]
            if len(match_shap) > 0 and shap_col in match_shap.columns:
                obs_shap = float(match_shap[shap_col].iloc[0])
            else:
                obs_shap = 0.0

            # Directional consistency rule:
            # If scenario reduces feature signal (e.g. biomass_low, wind_stagnant), positive SHAP should yield negative delta.
            if "low" in scenario or "q25" in scenario:
                if obs_shap > 0 and delta < 0:
                    consistent = True
                    conflict = False
                elif obs_shap < 0 and delta > 0:
                    consistent = True
                    conflict = False
                else:
                    consistent = False
                    conflict = True
            else:
                consistent = True
                conflict = False

            rows.append({
                "date": dt,
                "scenario": scenario,
                "target_group": grp,
                "observed_shap_value": obs_shap,
                "delta_prediction": delta,
                "directional_consistency": consistent,
                "counterfactual_shap_conflict": conflict
            })

        consistency_df = pd.DataFrame(rows)
        consistency_df.to_csv(self.exp_dir / "shap_counterfactual_consistency.csv", index=False)

        consistent_pct = float(np.mean(consistency_df["directional_consistency"])) * 100
        logger.info(f"SHAP-Counterfactual Consistency evaluation complete -> {consistent_pct:.1f}% directional consistency.")

        return consistency_df


if __name__ == "__main__":
    evaluator = SHAPConsistencyPhase4D()
