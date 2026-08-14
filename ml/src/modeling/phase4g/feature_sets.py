import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def get_feature_sets(v3_df: pd.DataFrame):
    """
    Returns controlled feature sets for Phase 4G incremental information experiments.
    """
    v2_registry_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "feature_registry.csv"
    v2_reg = pd.read_csv(v2_registry_path)
    base_v2_features = list(v2_reg['feature_name'].values)

    # Ensure all base features exist in v3_df
    base_v2_features = [f for f in base_v2_features if f in v3_df.columns]

    rainfall_features = ["rainfall_1d", "rainfall_3d", "rainfall_7d", "rain_event_1d", "washout_index_3d"]
    pbl_features = ["pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"]
    wind_features = ["wind_u_component_1d", "wind_v_component_1d", "upwind_stubble_quadrant_1d"]
    aod_features = ["aod_550_1d", "aod_roll_mean_3d"]

    feature_sets = {
        "Set_A_Baseline_V2": base_v2_features,
        "Set_B_V2_Rainfall": base_v2_features + rainfall_features,
        "Set_C_V2_Rainfall_PBL": base_v2_features + rainfall_features + pbl_features,
        "Set_D_V2_Rainfall_PBL_Winds": base_v2_features + rainfall_features + pbl_features + wind_features,
        "Set_E_All_External_Groups": base_v2_features + rainfall_features + pbl_features + wind_features + aod_features
    }

    return feature_sets
