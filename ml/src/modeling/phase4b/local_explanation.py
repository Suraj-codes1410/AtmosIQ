import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("LocalExplanationPhase4B")


class LocalExplanationEnginePhase4B:
    """
    AtmosIQ Phase 4B Individual Date Local Explanation API & Representative Date Selector.
    Provides explain_date("YYYY-MM-DD") API and selects 5 representative dates for waterfall plots.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4b"):
        self.exp_dir = Path(exp_dir)

    def explain_date(self, target_date: str, df: pd.DataFrame, feature_order: list, feature_to_group: dict, shap_matrix: np.ndarray, base_value: float, group_shap_df: pd.DataFrame) -> dict:
        """Explains model prediction for a specific date YYYY-MM-DD."""
        row_idx = df[df["date"] == target_date].index
        if len(row_idx) == 0:
            raise ValueError(f"Date '{target_date}' not found in Dataset v2!")

        idx = row_idx[0]
        act_pm25 = float(df.loc[idx, "pm25"])
        pred_pm25 = float(group_shap_df.loc[idx, "predicted_pm25"])

        # Feature level contributions
        feat_contribs = []
        for j, f_name in enumerate(feature_order):
            f_val = float(df.loc[idx, f_name])
            s_val = float(shap_matrix[idx, j])
            grp = feature_to_group.get(f_name, "unmapped")
            feat_contribs.append({
                "feature_name": f_name,
                "feature_value": f_val,
                "shap_value": s_val,
                "abs_shap_value": abs(s_val),
                "attribution_group": grp
            })

        feat_contribs_df = pd.DataFrame(feat_contribs).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

        # Group level contributions
        grp_cols = [c for c in group_shap_df.columns if c.endswith("_shap")]
        group_contribs = []
        for c in grp_cols:
            grp_name = c.replace("_shap", "")
            g_val = float(group_shap_df.loc[idx, c])
            group_contribs.append({
                "attribution_group": grp_name,
                "group_shap_value": g_val,
                "abs_group_shap_value": abs(g_val)
            })

        group_contribs_df = pd.DataFrame(group_contribs).sort_values("abs_group_shap_value", ascending=False).reset_index(drop=True)

        return {
            "date": target_date,
            "actual_pm25": act_pm25,
            "predicted_pm25": pred_pm25,
            "base_value": base_value,
            "feature_contributions": feat_contribs_df,
            "group_contributions": group_contribs_df,
            "row_index": idx
        }

    def select_representative_dates(self, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> dict:
        """Deterministically selects 5 representative dates for local waterfall plots."""
        test_df = df[df["date"].str.startswith("2024")].copy()
        test_indices = test_df.index.tolist()

        # 1. Low PM2.5 day in test set
        low_idx = test_df["pm25"].idxmin()
        low_date = test_df.loc[low_idx, "date"]

        # 2. Median PM2.5 day in test set
        med_pm25 = test_df["pm25"].median()
        med_idx = (test_df["pm25"] - med_pm25).abs().idxmin()
        med_date = test_df.loc[med_idx, "date"]

        # 3. High PM2.5 day in test set
        high_idx = test_df["pm25"].idxmax()
        high_date = test_df.loc[high_idx, "date"]

        # 4. Major post-monsoon episode (November 2024 stubble peak episode)
        nov_2024 = test_df[test_df["date"].str.startswith("2024-11")]
        episode_idx = nov_2024["pm25"].idxmax()
        episode_date = nov_2024.loc[episode_idx, "date"]

        # 5. Model failure / highest absolute residual day in test set
        residuals = (test_df["pm25"] - group_shap_df.loc[test_indices, "predicted_pm25"]).abs()
        fail_idx = residuals.idxmax()
        fail_date = test_df.loc[fail_idx, "date"]

        rep_dates = {
            "low_pm25": low_date,
            "median_pm25": med_date,
            "high_pm25": high_date,
            "episode_post_monsoon": episode_date,
            "high_residual_failure": fail_date
        }

        logger.info(f"Selected 5 Representative Dates for Local Waterfall Plots: {rep_dates}")
        return rep_dates


if __name__ == "__main__":
    local_api = LocalExplanationEnginePhase4B()
