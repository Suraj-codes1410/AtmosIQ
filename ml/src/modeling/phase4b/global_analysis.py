import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("GlobalAnalysisPhase4B")


class GlobalAnalysisPhase4B:
    """
    AtmosIQ Phase 4B Global Feature & Group Importance Analyzer.
    Computes global feature rankings, group SHAP rankings, quantile-based high-pollution day comparison, and flags extreme caution cases.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4b"):
        self.exp_dir = Path(exp_dir)
        self.summary_dir = self.exp_dir / "summaries"
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def analyze_global_importance(self, feature_order: list, feature_to_group: dict, shap_matrix: np.ndarray, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> dict:
        """Computes global feature, group, high-pollution, and extreme caution metrics."""
        logger.info("Computing Global Feature & Group Importance Summaries...")

        # 1. Global Feature Importance
        feat_rows = []
        for j, f_name in enumerate(feature_order):
            s_vals = shap_matrix[:, j]
            mean_abs = float(np.mean(np.abs(s_vals)))
            mean_signed = float(np.mean(s_vals))
            med_abs = float(np.median(np.abs(s_vals)))
            grp = feature_to_group.get(f_name, "unmapped")

            feat_rows.append({
                "feature_name": f_name,
                "attribution_group": grp,
                "mean_abs_shap": mean_abs,
                "mean_shap": mean_signed,
                "median_abs_shap": med_abs
            })

        feat_imp_df = pd.DataFrame(feat_rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        feat_imp_df["rank"] = feat_imp_df.index + 1
        feat_imp_df.to_csv(self.summary_dir / "global_feature_importance.csv", index=False)

        # 2. Global Group Importance
        ordered_groups = ["pm25_persistence", "meteorology", "wind_ventilation", "biomass_burning", "calendar_seasonal"]
        grp_rows = []
        for grp in ordered_groups:
            col_name = f"{grp}_shap"
            if col_name in group_shap_df.columns:
                g_vals = group_shap_df[col_name].values
                mean_abs = float(np.mean(np.abs(g_vals)))
                mean_signed = float(np.mean(g_vals))
                med_abs = float(np.median(np.abs(g_vals)))

                grp_rows.append({
                    "attribution_group": grp,
                    "mean_abs_shap": mean_abs,
                    "mean_signed_shap": mean_signed,
                    "median_abs_shap": med_abs
                })

        grp_imp_df = pd.DataFrame(grp_rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        grp_imp_df["rank"] = grp_imp_df.index + 1
        grp_imp_df.to_csv(self.summary_dir / "global_group_importance.csv", index=False)

        # 3. High-Pollution Days Analysis (Top 10% observed PM2.5, >= 90th percentile)
        p90_val = float(df["pm25"].quantile(0.90))
        logger.info(f"High-Pollution 90th Percentile Threshold: {p90_val:.2f} µg/m³.")

        high_mask = df["pm25"] >= p90_val
        high_group_df = group_shap_df[high_mask]
        normal_group_df = group_shap_df[~high_mask]

        high_analysis_rows = []
        for grp in ordered_groups:
            col_name = f"{grp}_shap"
            if col_name in group_shap_df.columns:
                h_mean_signed = float(high_group_df[col_name].mean())
                n_mean_signed = float(normal_group_df[col_name].mean())
                h_mean_abs = float(high_group_df[col_name].abs().mean())
                n_mean_abs = float(normal_group_df[col_name].abs().mean())

                high_analysis_rows.append({
                    "attribution_group": grp,
                    "high_pollution_mean_shap": h_mean_signed,
                    "normal_days_mean_shap": n_mean_signed,
                    "high_pollution_mean_abs_shap": h_mean_abs,
                    "normal_days_mean_abs_shap": n_mean_abs,
                    "high_pollution_delta_shap": h_mean_signed - n_mean_signed,
                    "high_pollution_percentile_group": "top_10_percent_p90"
                })

        high_analysis_df = pd.DataFrame(high_analysis_rows).sort_values("high_pollution_mean_shap", ascending=False)
        high_analysis_df.to_csv(self.summary_dir / "high_pollution_analysis.csv", index=False)

        # 4. Out-of-Distribution / Extreme Residual Day Flagging (Section 24)
        residuals = df["pm25"].values - group_shap_df["predicted_pm25"].values
        abs_residuals = np.abs(residuals)
        p95_res = float(np.percentile(abs_residuals, 95))

        extreme_mask = abs_residuals >= p95_res
        extreme_df = df[extreme_mask].copy()
        extreme_df["residual"] = residuals[extreme_mask]
        extreme_df["abs_residual"] = abs_residuals[extreme_mask]
        extreme_df["predicted_pm25"] = group_shap_df.loc[extreme_mask, "predicted_pm25"].values
        extreme_df["caution_flag"] = "attribution_caution_case_high_residual"

        extreme_df = extreme_df[["date", "pm25", "predicted_pm25", "residual", "abs_residual", "caution_flag"]].sort_values("abs_residual", ascending=False)
        extreme_df.to_csv(self.summary_dir / "extreme_caution_cases.csv", index=False)

        logger.info(f"Global analysis complete. Top Feature: {feat_imp_df.iloc[0]['feature_name']} (Mean Abs SHAP: {feat_imp_df.iloc[0]['mean_abs_shap']:.4f} µg/m³).")
        logger.info(f"Top Group: {grp_imp_df.iloc[0]['attribution_group']} (Mean Abs SHAP: {grp_imp_df.iloc[0]['mean_abs_shap']:.4f} µg/m³).")

        return {
            "feature_importance": feat_imp_df,
            "group_importance": grp_imp_df,
            "high_pollution_analysis": high_analysis_df,
            "extreme_caution_cases": extreme_df,
            "p90_threshold": p90_val
        }


if __name__ == "__main__":
    analyzer = GlobalAnalysisPhase4B()
