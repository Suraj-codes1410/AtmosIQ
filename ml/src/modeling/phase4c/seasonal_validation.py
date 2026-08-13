import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("SeasonalValidationPhase4C")


class SeasonalValidationPhase4C:
    """
    AtmosIQ Phase 4C Seasonal Attribution Validator.
    Evaluates group SHAP attributions across Winter, Summer, Monsoon, and Post-Monsoon regimes.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.season_mapping = {
            12: "winter", 1: "winter", 2: "winter",
            3: "summer", 4: "summer", 5: "summer",
            6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
            10: "post_monsoon", 11: "post_monsoon"
        }
        self.ordered_groups = [
            "pm25_persistence",
            "meteorology",
            "wind_ventilation",
            "biomass_burning",
            "calendar_seasonal"
        ]

    def validate_seasonal_patterns(self, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> pd.DataFrame:
        """Computes Season x Attribution Group matrix."""
        logger.info("Computing Season x Attribution Group validation matrix...")

        dates_dt = pd.to_datetime(df["date"])
        months = dates_dt.dt.month
        seasons = months.map(self.season_mapping)

        seasonal_rows = []
        for season_name in ["winter", "summer", "monsoon", "post_monsoon"]:
            s_mask = seasons == season_name
            s_group_df = group_shap_df[s_mask]

            # Rank groups within season
            mean_abs_dict = {grp: float(np.mean(np.abs(s_group_df[f"{grp}_shap"]))) for grp in self.ordered_groups}
            ranks = pd.Series(mean_abs_dict).rank(ascending=False).to_dict()

            for grp in self.ordered_groups:
                col_name = f"{grp}_shap"
                vals = s_group_df[col_name].values
                mean_abs = float(np.mean(np.abs(vals)))
                mean_signed = float(np.mean(vals))
                med_val = float(np.median(vals))
                p90_val = float(np.percentile(vals, 90))

                seasonal_rows.append({
                    "season": season_name,
                    "attribution_group": grp,
                    "mean_abs_shap": mean_abs,
                    "mean_signed_shap": mean_signed,
                    "median_shap": med_val,
                    "p90_shap": p90_val,
                    "group_rank_in_season": int(ranks[grp]),
                    "observation_count": int(s_mask.sum())
                })

        seasonal_df = pd.DataFrame(seasonal_rows)
        seasonal_df.to_csv(self.exp_dir / "seasonal_validation.csv", index=False)

        logger.info(f"Seasonal validation matrix created. Exported to {self.exp_dir / 'seasonal_validation.csv'}.")
        return seasonal_df


if __name__ == "__main__":
    validator = SeasonalValidationPhase4C()
