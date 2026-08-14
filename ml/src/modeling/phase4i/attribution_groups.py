import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("AttributionGroupsPhase4I")


class EnvironmentalAttributionGroupsPhase4I:
    """
    Maps the 35 Candidate_C_V3_Compact features into 6 environmental attribution groups.
    Generates v3_attribution_groups.csv.
    """

    GROUP_MAP = {
        # PM2.5 Persistence (10)
        "pm25_lag_1d": ("pm25_persistence", "Prior day PM2.5 baseline persistence", "Dataset_v3", 1.0),
        "pm25_lag_2d": ("pm25_persistence", "2-day lag PM2.5 memory", "Dataset_v3", 1.0),
        "pm25_lag_3d": ("pm25_persistence", "3-day lag PM2.5 memory", "Dataset_v3", 1.0),
        "pm25_lag_7d": ("pm25_persistence", "Weekly lag PM2.5 memory", "Dataset_v3", 1.0),
        "pm25_roll_mean_3d": ("pm25_persistence", "Short-term 3-day mean accumulation", "Dataset_v3", 1.0),
        "pm25_roll_mean_7d": ("pm25_persistence", "Medium-term 7-day mean accumulation", "Dataset_v3", 1.0),
        "pm25_roll_mean_14d": ("pm25_persistence", "Long-term 14-day mean accumulation", "Dataset_v3", 1.0),
        "pm25_roll_std_7d": ("pm25_persistence", "7-day PM2.5 volatility/variance", "Dataset_v3", 1.0),
        "pm25_roll_max_7d": ("pm25_persistence", "7-day peak PM2.5 exposure", "Dataset_v3", 1.0),
        "pm25_roll_min_7d": ("pm25_persistence", "7-day minimum baseline pollution", "Dataset_v3", 1.0),

        # Meteorology (6)
        "temperature_c_lag_1d": ("meteorology", "Ambient surface temperature lag", "Dataset_v3", 1.0),
        "temperature_c_roll_mean_3d": ("meteorology", "3-day thermal trend", "Dataset_v3", 1.0),
        "temperature_c_roll_min_3d": ("meteorology", "3-day minimum temperature / winter inversion indicator", "Dataset_v3", 1.0),
        "humidity_pct_lag_1d": ("meteorology", "Relative humidity lag", "Dataset_v3", 1.0),
        "humidity_pct_roll_mean_3d": ("meteorology", "3-day average humidity / hygroscopic growth", "Dataset_v3", 1.0),
        "humidity_pct_roll_max_7d": ("meteorology", "7-day maximum humidity indicator", "Dataset_v3", 1.0),

        # Wind & Ventilation (8)
        "wind_speed_kmh_lag_1d": ("wind_ventilation", "Surface wind speed lag", "Dataset_v3", 1.0),
        "wind_speed_kmh_roll_mean_3d": ("wind_ventilation", "3-day wind speed dispersion capability", "Dataset_v3", 1.0),
        "wind_u_component_1d": ("wind_ventilation", "Zonal wind vector component (East-West)", "Dataset_v3_ERA5", 1.0),
        "wind_v_component_1d": ("wind_ventilation", "Meridional wind vector component (North-South)", "Dataset_v3_ERA5", 1.0),
        "pblh_1d": ("wind_ventilation", "Planetary Boundary Layer Height (mixing layer depth)", "Dataset_v3_ERA5", 1.0),
        "pblh_min_1d": ("wind_ventilation", "Minimum overnight PBL height (nocturnal inversion depth)", "Dataset_v3_ERA5", 1.0),
        "pblh_roll_mean_3d": ("wind_ventilation", "3-day mean mixing depth", "Dataset_v3_ERA5", 1.0),
        "ventilation_index_1d": ("wind_ventilation", "Ventilation Index (Wind Speed * PBLH)", "Dataset_v3_ERA5", 1.0),

        # Biomass Burning (5)
        "is_stubble_season": ("biomass_burning", "Northwest agricultural stubble burning window indicator", "Dataset_v3", 1.0),
        "fire_hotspot_count_lag_1d": ("biomass_burning", "VIIRS/MODIS active fire hotspot count", "Dataset_v3", 1.0),
        "fire_hotspot_count_roll_mean_3d": ("biomass_burning", "3-day mean upstream fire activity", "Dataset_v3", 1.0),
        "fire_hotspot_count_roll_mean_7d": ("biomass_burning", "7-day cumulative regional fire activity", "Dataset_v3", 1.0),
        "upwind_stubble_quadrant_1d": ("biomass_burning", "Stubble fires in NW upwind transport corridor", "Dataset_v3_NASA_FIRMS", 1.0),

        # External Environmental (4)
        "rainfall_1d": ("external_environmental", "Daily cumulative precipitation (IMD/ERA5)", "Dataset_v3_ERA5", 1.0),
        "rainfall_3d": ("external_environmental", "3-day accumulated rainfall", "Dataset_v3_ERA5", 1.0),
        "rain_event_1d": ("external_environmental", "Precipitation binary indicator (Rain >= 1mm)", "Dataset_v3_ERA5", 1.0),
        "washout_index_3d": ("external_environmental", "Wet deposition aerosol washout index", "Dataset_v3_ERA5", 1.0),

        # AOD & Calendar (2)
        "aod_550_1d": ("external_environmental", "MODIS Satellite Aerosol Optical Depth at 550nm", "Dataset_v3_MODIS", 1.0),
        "festival_window": ("calendar_seasonal", "Diwali & festival anthropogenic surge window", "Dataset_v3", 1.0)
    }

    def generate_mapping(self, features_35: list, output_csv: Path) -> pd.DataFrame:
        logger.info("Generating Environmental Group Mapping for 35 Features...")
        records = []
        for feat in features_35:
            if feat in self.GROUP_MAP:
                grp, rat, src, conf = self.GROUP_MAP[feat]
            else:
                grp = "other"
                rat = "Unclassified feature"
                src = "Dataset_v3"
                conf = 0.5

            records.append({
                "feature": feat,
                "group": grp,
                "rationale": rat,
                "source": src,
                "mapping_confidence": conf
            })

        df_grp = pd.DataFrame(records)
        df_grp.to_csv(output_csv, index=False)
        logger.info(f"Group Mapping saved to {output_csv} ({len(df_grp)} features mapped).")
        return df_grp
