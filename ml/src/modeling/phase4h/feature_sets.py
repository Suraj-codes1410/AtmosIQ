import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureSetsPhase4H")


class FeatureSetManagerPhase4H:
    """
    Candidate Feature Set Manager for Phase 4H.
    Constructs and verifies prediction-safe feature sets:
    - Candidate_A_V2_Baseline: 147 frozen v2 prediction-safe features.
    - Candidate_B_V3_Expanded: v2 features + validated external environmental features (rainfall, PBL, ventilation, wind, AOD).
    - Candidate_C_V3_Compact: Compact interpretable subset (~35 features).
    """

    def __init__(self, v3_df: pd.DataFrame, approved_features: list = None):
        self.v3_df = v3_df
        self.v3_cols = set(v3_df.columns)
        self.approved_features = set(approved_features) if approved_features else None

        # Load v2 baseline registry
        v2_reg_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "feature_registry.csv"
        if v2_reg_path.exists():
            v2_reg = pd.read_csv(v2_reg_path)
            self.v2_features = v2_reg['feature_name'].tolist()
        else:
            self.v2_features = [c for c in v3_df.columns if "_lag_" in c or "_roll_" in c][:147]

    def get_feature_sets(self) -> dict:
        # 1. Candidate A: v2 Baseline (147 features)
        candidate_a = [f for f in self.v2_features if f in self.v3_cols]

        # 2. Candidate B: v3 Expanded (v2 + external environmental features)
        ext_features = [
            "rainfall_1d", "rainfall_3d", "rainfall_7d", "rain_event_1d", "washout_index_3d",
            "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d",
            "aod_550_1d", "aod_roll_mean_3d",
            "wind_u_component_1d", "wind_v_component_1d", "upwind_stubble_quadrant_1d"
        ]
        ext_validated = [f for f in ext_features if f in self.v3_cols]
        candidate_b = candidate_a + [f for f in ext_validated if f not in candidate_a]

        # 3. Candidate C: v3 Compact (35 key interpretable features)
        compact_candidates = [
            # Core PM2.5 Lags & Rolling
            "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
            "pm25_roll_mean_3d", "pm25_roll_mean_7d", "pm25_roll_mean_14d",
            "pm25_roll_std_7d", "pm25_roll_max_7d", "pm25_roll_min_7d",
            # Temperature Lags & Rolling
            "temperature_c_lag_1d", "temperature_c_roll_mean_3d", "temperature_c_roll_min_3d",
            # Humidity Lags & Rolling
            "humidity_pct_lag_1d", "humidity_pct_roll_mean_3d", "humidity_pct_roll_max_7d",
            # Wind Speed & Components
            "wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d", "wind_u_component_1d", "wind_v_component_1d",
            # Stubble & Fire Hotspots
            "is_stubble_season", "fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d",
            "fire_hotspot_count_roll_mean_7d", "upwind_stubble_quadrant_1d",
            # Rainfall & Washout
            "rainfall_1d", "rainfall_3d", "rain_event_1d", "washout_index_3d",
            # PBL & Boundary Layer
            "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d",
            # AOD & Calendar
            "aod_550_1d", "festival_window"
        ]
        candidate_c = [f for f in compact_candidates if f in self.v3_cols]

        feature_sets = {
            "Candidate_A_V2_Baseline": candidate_a,
            "Candidate_B_V3_Expanded": candidate_b,
            "Candidate_C_V3_Compact": candidate_c
        }

        # Filter against approved features if available
        if self.approved_features:
            for k in list(feature_sets.keys()):
                feature_sets[k] = [f for f in feature_sets[k] if f in self.approved_features]

        for k, v in feature_sets.items():
            logger.info(f"Feature Set '{k}': {len(v)} features.")

        return feature_sets
