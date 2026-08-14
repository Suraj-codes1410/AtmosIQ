import sys
from pathlib import Path
import json
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureRegistryV3")


class FeatureRegistryV3:
    """
    Feature Registry & Machine-Readable Data Dictionary for Dataset v3.
    """

    def build_registry(self, ext_cols: list, output_dir: Path) -> pd.DataFrame:
        logger.info("Building Dataset v3 Feature Registry & Schema Dictionary...")
        registry = []

        # External candidate features descriptions
        ext_metadata = {
            "rainfall_1d": ("precipitation", "IMD/ERA5", "mm/day", "Daily surface rainfall total", "1d", "0.25° grid", "t-1d", "Daily Sum", True, "precipitation_amount_mm"),
            "rainfall_3d": ("precipitation", "IMD/ERA5", "mm", "3-day cumulative rainfall total", "3d", "0.25° grid", "t-1d", "3-day Rolling Sum", True, "rainfall_1d"),
            "rainfall_7d": ("precipitation", "IMD/ERA5", "mm", "7-day cumulative rainfall total", "7d", "0.25° grid", "t-1d", "7-day Rolling Sum", True, "rainfall_1d"),
            "rain_event_1d": ("precipitation", "IMD/ERA5", "binary (0/1)", "Rainfall event occurrence flag (>=1.0mm)", "1d", "0.25° grid", "t-1d", "Threshold Indicator", True, "rainfall_1d"),
            "washout_index_3d": ("precipitation", "IMD/ERA5", "index", "Non-linear rain washout potential log(1+rainfall_3d)", "3d", "0.25° grid", "t-1d", "Log Transformation", True, "rainfall_3d"),
            "pblh_1d": ("pbl_height", "ERA5", "meters", "Daily mean Planetary Boundary Layer Height", "1d", "0.25° grid", "t-1d", "Daily Mean", True, "pbl_height_mean_m"),
            "pblh_min_1d": ("pbl_height", "ERA5", "meters", "Daily minimum Planetary Boundary Layer Height", "1d", "0.25° grid", "t-1d", "Daily Minimum", True, "pbl_height_min_m"),
            "pblh_roll_mean_3d": ("pbl_height", "ERA5", "meters", "3-day rolling mean Planetary Boundary Layer Height", "3d", "0.25° grid", "t-1d", "3-day Rolling Mean", True, "pblh_1d"),
            "ventilation_index_1d": ("pbl_height", "ERA5/CPCB", "m2/s", "Atmospheric ventilation index (PBLH * wind_speed)", "1d", "Delhi NCR Regional", "t-1d", "Product Interaction", True, "pblh_1d, wind_speed"),
            "aod_550_1d": ("aerosol", "NASA MODIS", "unitless", "Regional mean 550nm Aerosol Optical Depth", "1d", "1.0° grid", "t-1d", "Spatial Mean", True, "aod_550_mean"),
            "aod_roll_mean_3d": ("aerosol", "NASA MODIS", "unitless", "3-day rolling mean Aerosol Optical Depth", "3d", "1.0° grid", "t-1d", "3-day Rolling Mean", True, "aod_550_1d"),
            "wind_u_component_1d": ("transport_winds", "ERA5", "m/s", "East-West u-component transport wind", "1d", "0.25° grid", "t-1d", "Vector Component", True, "wind_speed, wind_dir"),
            "wind_v_component_1d": ("transport_winds", "ERA5", "m/s", "North-South v-component transport wind", "1d", "0.25° grid", "t-1d", "Vector Component", True, "wind_speed, wind_dir"),
            "upwind_stubble_quadrant_1d": ("transport_winds", "ERA5/IMD", "km/h", "NW upwind stubble transport quadrant index", "1d", "Delhi NCR Regional", "t-1d", "Quadrant Filter", True, "wind_speed, wind_dir"),
        }

        for col in ext_cols:
            if col in ext_metadata:
                cat, src, unit, desc, t_res, s_res, lag, agg, safe, derived = ext_metadata[col]
            else:
                cat, src, unit, desc, t_res, s_res, lag, agg, safe, derived = ("external", "Custom", "raw", "External feature", "1d", "NCR", "t-1d", "Direct", True, "raw")

            registry.append({
                "feature_name": col,
                "category": cat,
                "source": src,
                "unit": unit,
                "description": desc,
                "temporal_resolution": t_res,
                "spatial_resolution": s_res,
                "availability_lag": lag,
                "aggregation_method": agg,
                "leakage_safe": safe,
                "derived_from": derived,
                "license": "Open Data License",
                "provenance": "Phase 4G Ingestion Pipeline"
            })

        df_reg = pd.DataFrame(registry)
        csv_path = output_dir / "feature_registry_v3.csv"
        df_reg.to_csv(csv_path, index=False)
        logger.info(f"Feature Registry V3 CSV saved to {csv_path}.")

        # Generate Json Feature Set Registry
        feature_sets = {
            "Set_A_Baseline_V2": "147 prediction-safe features from Dataset v2",
            "Set_B_V2_Rainfall": ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rain_event_1d", "washout_index_3d"],
            "Set_C_V2_Rainfall_PBL": ["pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"],
            "Set_D_V2_Rainfall_PBL_Winds": ["wind_u_component_1d", "wind_v_component_1d", "upwind_stubble_quadrant_1d"],
            "Set_E_All_External_Groups": list(ext_cols)
        }
        json_path = output_dir / "feature_set_registry_v3.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(feature_sets, f, indent=2)

        return df_reg
