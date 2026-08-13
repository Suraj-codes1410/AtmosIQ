import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("TemporalAnalysisPhase4B")


class TemporalAnalysisPhase4B:
    """
    AtmosIQ Phase 4B Temporal & Seasonal Attribution Analyzer.
    Investigates group and feature SHAP behavior across seasons (Winter, Summer, Monsoon, Post-Monsoon) and evaluates multi-year temporal stability (2022-2024).
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4b"):
        self.exp_dir = Path(exp_dir)
        self.summary_dir = self.exp_dir / "summaries"
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.season_mapping = {
            12: "winter", 1: "winter", 2: "winter",
            3: "summer", 4: "summer", 5: "summer",
            6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
            10: "post_monsoon", 11: "post_monsoon"
        }

    def analyze_temporal_patterns(self, df: pd.DataFrame, group_shap_df: pd.DataFrame, shap_matrix: np.ndarray, feature_order: list) -> dict:
        """Executes seasonal breakdown and multi-year stability analysis."""
        logger.info("Executing Seasonal & Multi-Year Temporal Stability Analysis...")

        df_copy = df.copy()
        df_copy["date_dt"] = pd.to_datetime(df_copy["date"])
        df_copy["year"] = df_copy["date_dt"].dt.year
        df_copy["month"] = df_copy["date_dt"].dt.month
        df_copy["season"] = df_copy["month"].map(self.season_mapping)

        # 1. Group SHAP by Season
        grp_cols = [c for c in group_shap_df.columns if c.endswith("_shap")]
        merged_group_df = pd.concat([df_copy[["date", "year", "month", "season"]], group_shap_df[grp_cols]], axis=1)

        seasonal_group_summary = merged_group_df.groupby("season")[grp_cols].mean().reset_index()
        seasonal_group_summary.to_csv(self.summary_dir / "seasonal_group_summary.csv", index=False)

        # 2. Multi-Year Feature Rank Stability (2022, 2023, 2024)
        top10_by_year = {}
        years = [2022, 2023, 2024]

        for yr in years:
            yr_mask = df_copy["year"] == yr
            if yr_mask.sum() > 0:
                yr_shap = shap_matrix[yr_mask]
                mean_abs = np.mean(np.abs(yr_shap), axis=0)
                top_idx = np.argsort(mean_abs)[::-1][:10]
                top10_by_year[yr] = set([feature_order[i] for i in top_idx])

        # Overlap metrics
        overlap_22_23 = len(top10_by_year[2022].intersection(top10_by_year[2023])) / 10.0 if 2022 in top10_by_year and 2023 in top10_by_year else 0.0
        overlap_23_24 = len(top10_by_year[2023].intersection(top10_by_year[2024])) / 10.0 if 2023 in top10_by_year and 2024 in top10_by_year else 0.0
        overlap_22_24 = len(top10_by_year[2022].intersection(top10_by_year[2024])) / 10.0 if 2022 in top10_by_year and 2024 in top10_by_year else 0.0

        stability_rows = [
            {"period_comparison": "2022 vs 2023", "top10_feature_overlap": overlap_22_23, "stability_status": "STABLE" if overlap_22_23 >= 0.7 else "MODERATE"},
            {"period_comparison": "2023 vs 2024", "top10_feature_overlap": overlap_23_24, "stability_status": "STABLE" if overlap_23_24 >= 0.7 else "MODERATE"},
            {"period_comparison": "2022 vs 2024", "top10_feature_overlap": overlap_22_24, "stability_status": "STABLE" if overlap_22_24 >= 0.7 else "MODERATE"}
        ]

        stability_df = pd.DataFrame(stability_rows)
        stability_df.to_csv(self.summary_dir / "temporal_stability.csv", index=False)

        logger.info(f"Temporal Stability complete. 2023 vs 2024 Top-10 Feature Overlap: {overlap_23_24 * 100:.1f}%.")

        return {
            "seasonal_summary": seasonal_group_summary,
            "temporal_stability": stability_df,
            "merged_group_df": merged_group_df
        }


if __name__ == "__main__":
    analyzer = TemporalAnalysisPhase4B()
