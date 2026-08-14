import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ExtremeSeasonalStabilityPhase4I")


class ExtremeSeasonalStabilityEnginePhase4I:
    """
    Extreme Pollution, Seasonal, and Multi-Year Stability Analysis Engine for Phase 4I.
    Evaluates PM2.5 >= 150 ug/m3, 4 seasons, and 2020-2024 temporal stability.
    """

    SEASONS = {
        "Winter": [12, 1, 2],
        "Summer": [3, 4, 5],
        "Monsoon": [6, 7, 8, 9],
        "Post-Monsoon": [10, 11]
    }

    def __init__(self, df_v3: pd.DataFrame, df_shap_all: pd.DataFrame, df_group_shap_all: pd.DataFrame, features_35: list):
        self.df_v3 = df_v3.copy()
        self.df_v3['date'] = pd.to_datetime(self.df_v3['date'])
        self.df_shap_all = df_shap_all
        self.df_group_shap_all = df_group_shap_all
        self.features = features_35

    def run_all(self, output_dir: Path) -> dict:
        logger.info("Executing Extreme Event, Seasonal, and Multi-Year Stability Analysis...")
        output_dir.mkdir(parents=True, exist_ok=True)

        actuals = self.df_v3['pm25'].values
        preds = self.df_group_shap_all['predicted_pm25'].values
        months = self.df_v3['date'].dt.month.values
        years = self.df_v3['date'].dt.year.values

        # 1. Extreme Pollution Analysis (PM2.5 >= 150 ug/m3)
        extreme_mask = (actuals >= 150.0)
        top10_threshold = float(np.percentile(actuals, 90))
        top10_mask = (actuals >= top10_threshold)

        extreme_records = []
        for label, mask in [("PM25_ge_150", extreme_mask), ("Top_10_Percentile", top10_mask)]:
            if mask.sum() > 0:
                y_sub = actuals[mask]
                p_sub = preds[mask]
                mae = float(np.mean(np.abs(y_sub - p_sub)))
                rmse = float(np.sqrt(np.mean((y_sub - p_sub)**2)))

                # Group SHAP means during extreme events
                grp_cols = [c for c in self.df_group_shap_all.columns if c not in ['date', 'year', 'actual_pm25', 'predicted_pm25']]
                grp_means = self.df_group_shap_all.loc[mask, grp_cols].mean().to_dict()

                rec = {
                    "subset": label,
                    "count_days": int(mask.sum()),
                    "mae_ugm3": mae,
                    "rmse_ugm3": rmse
                }
                for g, v in grp_means.items():
                    rec[f"group_mean_shap_{g}"] = float(v)
                extreme_records.append(rec)

        df_extreme = pd.DataFrame(extreme_records)
        df_extreme.to_csv(output_dir / "v3_extreme_event_analysis.csv", index=False)

        # 2. Seasonal Analysis
        seasonal_records = []
        for season_name, season_months in self.SEASONS.items():
            s_mask = np.isin(months, season_months)
            if s_mask.sum() > 0:
                y_s = actuals[s_mask]
                p_s = preds[s_mask]
                mae_s = float(np.mean(np.abs(y_s - p_s)))
                rmse_s = float(np.sqrt(np.mean((y_s - p_s)**2)))

                # Top SHAP feature in season
                feat_cols = [c for c in self.df_shap_all.columns if c not in ['date', 'year', 'actual_pm25', 'predicted_pm25', 'base_value']]
                feat_s_means = self.df_shap_all.loc[s_mask, feat_cols].abs().mean()
                top_feat = feat_s_means.idxmax()

                grp_cols = [c for c in self.df_group_shap_all.columns if c not in ['date', 'year', 'actual_pm25', 'predicted_pm25']]
                grp_s_means = self.df_group_shap_all.loc[s_mask, grp_cols].abs().mean()
                top_grp = grp_s_means.idxmax()

                rec_s = {
                    "season": season_name,
                    "count_days": int(s_mask.sum()),
                    "mean_mae_ugm3": mae_s,
                    "mean_rmse_ugm3": rmse_s,
                    "top_group": top_grp,
                    "top_feature": top_feat
                }
                for g in grp_cols:
                    rec_s[f"group_mean_abs_shap_{g}"] = float(grp_s_means[g])
                seasonal_records.append(rec_s)

        df_seasonal = pd.DataFrame(seasonal_records)
        df_seasonal.to_csv(output_dir / "v3_seasonal_analysis.csv", index=False)

        # 3. Multi-Year Stability Analysis (2020-2024)
        unique_yrs = sorted(list(set(years)))
        yearly_ranks = {}
        grp_cols = [c for c in self.df_group_shap_all.columns if c not in ['date', 'year', 'actual_pm25', 'predicted_pm25']]

        yearly_records = []
        for yr in unique_yrs:
            y_mask = (years == yr)
            if y_mask.sum() > 0:
                grp_yr_means = self.df_group_shap_all.loc[y_mask, grp_cols].abs().mean()
                sorted_grps = grp_yr_means.sort_values(ascending=False).index.tolist()
                yearly_ranks[yr] = sorted_grps

                yearly_records.append({
                    "year": yr,
                    "count_days": int(y_mask.sum()),
                    "top_group": sorted_grps[0],
                    "second_group": sorted_grps[1] if len(sorted_grps) > 1 else "NONE",
                    "top_feature": self.df_shap_all.loc[y_mask, feat_cols].abs().mean().idxmax()
                })

        # Calculate year-to-year rank correlations
        rhos = []
        for i in range(len(unique_yrs) - 1):
            y1, y2 = unique_yrs[i], unique_yrs[i+1]
            r1 = [grp_cols.index(g) for g in yearly_ranks[y1]]
            r2 = [grp_cols.index(g) for g in yearly_ranks[y2]]
            rho, _ = spearmanr(r1, r2)
            rhos.append(rho)

        avg_rho = float(np.mean(rhos)) if rhos else 1.0

        df_stab = pd.DataFrame(yearly_records)
        df_stab["mean_year_to_year_spearman_rho"] = avg_rho
        df_stab["overall_multi_year_stability"] = "HIGH" if avg_rho >= 0.8 else "MODERATE"
        df_stab.to_csv(output_dir / "v3_temporal_stability.csv", index=False)

        logger.info("Extreme event, seasonal, and multi-year stability analysis complete.")
        return {
            "df_extreme": df_extreme,
            "df_seasonal": df_seasonal,
            "df_stab": df_stab
        }
