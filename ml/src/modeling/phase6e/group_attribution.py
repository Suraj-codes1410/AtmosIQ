import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6e.config import InterpretabilityConfigPhase6E

logger = setup_logger("GroupAttributionPhase6E")


class GroupAttributionEnginePhase6E:
    """
    Environmental Group-Level Attribution Uncertainty Engine for Phase 6E.
    Aggregates feature SHAP values into 6 environmental groups and computes group-level dispersion.
    """

    GROUP_DEFINITIONS = {
        "pm25_persistence": [
            "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
            "pm25_roll_mean_3d", "pm25_roll_mean_7d", "pm25_roll_mean_14d",
            "pm25_roll_std_7d", "pm25_roll_max_7d", "pm25_roll_min_7d"
        ],
        "meteorology": [
            "temperature_c_lag_1d", "temperature_c_roll_mean_3d", "temperature_c_roll_min_3d",
            "humidity_pct_lag_1d", "humidity_pct_roll_mean_3d", "humidity_pct_roll_max_7d"
        ],
        "wind_ventilation": [
            "wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d",
            "wind_u_component_1d", "wind_v_component_1d",
            "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"
        ],
        "biomass_burning": [
            "is_stubble_season", "fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d",
            "fire_hotspot_count_roll_mean_7d", "upwind_stubble_quadrant_1d"
        ],
        "external_environmental": [
            "rainfall_1d", "rainfall_3d", "rain_event_1d", "washout_index_3d", "aod_550_1d"
        ],
        "calendar_seasonal": [
            "festival_window"
        ]
    }

    def __init__(self, config: InterpretabilityConfigPhase6E):
        self.config = config

    def run_group_attribution_analysis(
        self,
        df_shap_obs: pd.DataFrame,
        output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Executing Environmental Group-Level Attribution Uncertainty Aggregation...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map each feature to its group
        feat_to_group = {}
        for g_name, f_list in self.GROUP_DEFINITIONS.items():
            for f in f_list:
                feat_to_group[f] = g_name

        df_shap_obs['feature_group'] = df_shap_obs['feature_name'].map(feat_to_group).fillna("other")

        # Group by date, eval_fold, year, pollution_regime, season, feature_group
        group_obs_records = []
        for (d_str, grp_name), sub_df in df_shap_obs.groupby(['date', 'feature_group']):
            row_meta = sub_df.iloc[0]
            
            # Sum the mean/median/percentiles across features in this group
            # Because sum of means is the mean of sum:
            grp_mean_shap = float(sub_df['mean_shap'].sum())
            grp_signed_std = float(np.sqrt(np.sum(sub_df['std_shap'] ** 2)))  # conservative dispersion estimate
            grp_q10 = float(sub_df['q10_shap'].sum())
            grp_q90 = float(sub_df['q90_shap'].sum())
            grp_q05 = float(sub_df['q05_shap'].sum())
            grp_q95 = float(sub_df['q95_shap'].sum())
            grp_sign_stab = float(sub_df['sign_stability'].mean())

            group_obs_records.append({
                "date": d_str,
                "year": int(row_meta['year']),
                "eval_fold": int(row_meta['eval_fold']),
                "pollution_regime": row_meta['pollution_regime'],
                "season": row_meta['season'],
                "feature_group": grp_name,
                "observed_pm25": float(row_meta['observed_pm25']),
                "mean_group_shap": grp_mean_shap,
                "std_group_shap": grp_signed_std,
                "q05_group_shap": grp_q05,
                "q10_group_shap": grp_q10,
                "q90_group_shap": grp_q90,
                "q95_group_shap": grp_q95,
                "attr_interval_80_lower": grp_q10,
                "attr_interval_80_upper": grp_q90,
                "attr_interval_90_lower": grp_q05,
                "attr_interval_90_upper": grp_q95,
                "group_sign_stability": grp_sign_stab
            })

        df_grp_obs = pd.DataFrame(group_obs_records)
        df_grp_obs.to_csv(output_dir / "group_attribution_uncertainty.csv", index=False)

        # Global Group-level summary
        grp_summary_records = []
        for grp_name in sorted(self.GROUP_DEFINITIONS.keys()):
            sub = df_grp_obs[df_grp_obs['feature_group'] == grp_name]
            mean_abs_attr = float(sub['mean_group_shap'].abs().mean())
            mean_signed_attr = float(sub['mean_group_shap'].mean())
            mean_std_attr = float(sub['std_group_shap'].mean())
            mean_sign_stab = float(sub['group_sign_stability'].mean())
            feature_count = len(self.GROUP_DEFINITIONS[grp_name])

            grp_summary_records.append({
                "feature_group": grp_name,
                "feature_count": feature_count,
                "mean_absolute_group_shap": mean_abs_attr,
                "mean_signed_group_shap": mean_signed_attr,
                "mean_group_shap_std": mean_std_attr,
                "group_sign_stability": mean_sign_stab,
                "q10_group_shap_mean": float(sub['q10_group_shap'].mean()),
                "q90_group_shap_mean": float(sub['q90_group_shap'].mean())
            })

        df_grp_summary = pd.DataFrame(grp_summary_records).sort_values("mean_absolute_group_shap", ascending=False).reset_index(drop=True)
        logger.info(f"Group attribution summary complete across {len(df_grp_summary)} environmental groups.")
        return df_grp_obs, df_grp_summary
