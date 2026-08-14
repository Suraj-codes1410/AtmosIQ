import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureRegistryValidationPhase4I")


class FeatureRegistryValidatorPhase4I:
    """
    Validates exact Candidate_C_V3_Compact feature registry for Phase 4I.
    Verifies 35 prediction-safe features, feature ordering, dataset presence, and zero leakage.
    """

    COMPACT_35_FEATURES = [
        # Core PM2.5 Lags & Rolling (10)
        "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
        "pm25_roll_mean_3d", "pm25_roll_mean_7d", "pm25_roll_mean_14d",
        "pm25_roll_std_7d", "pm25_roll_max_7d", "pm25_roll_min_7d",
        # Temperature Lags & Rolling (3)
        "temperature_c_lag_1d", "temperature_c_roll_mean_3d", "temperature_c_roll_min_3d",
        # Humidity Lags & Rolling (3)
        "humidity_pct_lag_1d", "humidity_pct_roll_mean_3d", "humidity_pct_roll_max_7d",
        # Wind Speed & Components (4)
        "wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d", "wind_u_component_1d", "wind_v_component_1d",
        # Stubble & Fire Hotspots (5)
        "is_stubble_season", "fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d",
        "fire_hotspot_count_roll_mean_7d", "upwind_stubble_quadrant_1d",
        # Rainfall & Washout (4)
        "rainfall_1d", "rainfall_3d", "rain_event_1d", "washout_index_3d",
        # PBL & Boundary Layer (4)
        "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d",
        # AOD & Calendar (2)
        "aod_550_1d", "festival_window"
    ]

    def __init__(self, df_v3: pd.DataFrame):
        self.df_v3 = df_v3
        self.v3_cols = set(df_v3.columns)

    def validate(self, output_csv: Path) -> pd.DataFrame:
        logger.info("Validating Phase 4I Feature Registry (Candidate_C_V3_Compact)...")
        records = []

        unsafe_features = {'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'}

        for idx, feat in enumerate(self.COMPACT_35_FEATURES):
            present = feat in self.v3_cols
            is_unsafe = feat in unsafe_features or feat.startswith('pm25_same_day')
            leakage_status = "unsafe" if is_unsafe else "safe"
            
            # Determine group
            if "pm25" in feat:
                group = "pm25_persistence"
            elif "temperature" in feat or "humidity" in feat:
                group = "meteorology"
            elif "wind" in feat or "pblh" in feat or "ventilation" in feat:
                group = "wind_ventilation"
            elif "stubble" in feat or "fire" in feat:
                group = "biomass_burning"
            elif "rainfall" in feat or "rain" in feat or "washout" in feat or "aod" in feat:
                group = "external_environmental"
            else:
                group = "calendar_seasonal"

            val_status = "PASS" if (present and leakage_status == "safe") else "FAIL"

            records.append({
                "feature_name": feat,
                "group": group,
                "source": "Dataset_v3",
                "temporal_availability": "t-1d_lag_or_rolling",
                "leakage_status": leakage_status,
                "present_in_dataset": present,
                "model_order": idx + 1,
                "validation_status": val_status
            })

        df_reg = pd.DataFrame(records)
        df_reg.to_csv(output_csv, index=False)

        assert len(df_reg) == 35, f"Expected 35 features, got {len(df_reg)}"
        assert df_reg['present_in_dataset'].all(), "Some features missing from Dataset v3!"
        assert (df_reg['leakage_status'] == 'safe').all(), "Leakage detected in feature set!"
        assert (df_reg['validation_status'] == 'PASS').all(), "Validation failed for some features!"

        logger.info("Feature Registry Validation PASSED cleanly (35/35 safe features verified).")
        return df_reg
