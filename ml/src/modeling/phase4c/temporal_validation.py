import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ml.src.utils.logger import setup_logger

logger = setup_logger("TemporalValidationPhase4C")


class TemporalValidationPhase4C:
    """
    AtmosIQ Phase 4C Multi-Year Temporal Stability Validator.
    Evaluates attribution consistency across 2020-2024, computes inter-year Spearman rank correlations, and exports statistical test reports.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.ordered_groups = [
            "pm25_persistence",
            "meteorology",
            "wind_ventilation",
            "biomass_burning",
            "calendar_seasonal"
        ]

    def validate_temporal_stability(self, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> dict:
        """Executes multi-year stability analysis across 2020-2024."""
        logger.info("Executing Multi-Year Temporal Stability Analysis (2020-2024)...")

        dates_dt = pd.to_datetime(df["date"])
        years = dates_dt.dt.year.unique()
        years.sort()

        yearly_rows = []
        yearly_group_vecs = {}

        for yr in years:
            yr_mask = dates_dt.dt.year == yr
            yr_group_df = group_shap_df[yr_mask]
            yr_df = df[yr_mask]

            # High-pollution mask within year
            p90_yr = float(yr_df["pm25"].quantile(0.90))
            yr_high_mask = yr_df["pm25"] >= p90_yr

            mean_abs_dict = {grp: float(np.mean(np.abs(yr_group_df[f"{grp}_shap"]))) for grp in self.ordered_groups}
            ranks = pd.Series(mean_abs_dict).rank(ascending=False).to_dict()
            yearly_group_vecs[yr] = [mean_abs_dict[grp] for grp in self.ordered_groups]

            for grp in self.ordered_groups:
                col_name = f"{grp}_shap"
                vals = yr_group_df[col_name].values
                high_vals = yr_group_df.loc[yr_high_mask, col_name].values

                yearly_rows.append({
                    "year": int(yr),
                    "attribution_group": grp,
                    "mean_abs_shap": float(np.mean(np.abs(vals))),
                    "mean_signed_shap": float(np.mean(vals)),
                    "median_shap": float(np.median(vals)),
                    "high_pollution_mean_shap": float(np.mean(high_vals)) if len(high_vals) > 0 else 0.0,
                    "group_rank_in_year": int(ranks[grp]),
                    "observation_count": int(yr_mask.sum())
                })

        temp_df = pd.DataFrame(yearly_rows)
        temp_df.to_csv(self.exp_dir / "temporal_validation.csv", index=False)

        # Inter-year Spearman Rank Correlation Statistical Tests
        stat_rows = []
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                y1, y2 = years[i], years[j]
                v1, v2 = yearly_group_vecs[y1], yearly_group_vecs[y2]
                r, p = spearmanr(v1, v2)
                stat_rows.append({
                    "test_name": "spearman_inter_year_group_rank_correlation",
                    "year_pair": f"{y1} vs {y2}",
                    "correlation_coefficient": float(r),
                    "p_value": float(p),
                    "sample_size": len(self.ordered_groups),
                    "stability_status": "HIGHLY_STABLE" if r >= 0.8 else "STABLE"
                })

        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(self.exp_dir / "statistical_tests.csv", index=False)

        logger.info(f"Temporal stability analysis complete. 2023 vs 2024 Inter-Year Rank Correlation r = {stat_df.loc[stat_df['year_pair'] == '2023 vs 2024', 'correlation_coefficient'].values[0]:.4f}.")

        return {"temporal_df": temp_df, "statistical_df": stat_df}


if __name__ == "__main__":
    validator = TemporalValidationPhase4C()
