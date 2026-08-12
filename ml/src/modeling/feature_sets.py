import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureSetsPhase3D")


class FeatureSetManager:
    """
    AtmosIQ Phase 3D: Feature Set Manager.
    Loads established Phase 3C candidate feature sets and constructs the exploratory
    set_b_plus_core_environment set with zero leakage.
    """

    def __init__(self, modeling_dir: str = "ml/data/modeling/v1"):
        self.modeling_dir = Path(modeling_dir)
        self.avail_path = self.modeling_dir / "feature_availability.csv"

    def load_safe_features(self) -> list[str]:
        """Loads authoritative safe feature whitelist."""
        assert self.avail_path.exists(), f"Missing feature availability file: {self.avail_path}"
        df_avail = pd.read_csv(self.avail_path)
        safe_cols = df_avail[df_avail["prediction_safe"] == True]["feature_name"].tolist()
        return safe_cols

    def get_phase3d_feature_sets(self) -> dict[str, list[str]]:
        """Returns the three candidate feature sets for Phase 3D tuning."""
        safe_cols = set(self.load_safe_features())

        # 1. Primary Feature Set: set_b_pm25_history (29 features)
        reg_file = Path("ml/experiments/phase3c/feature_set_registry.json")
        assert reg_file.exists(), f"Missing Phase 3C registry: {reg_file}"

        with open(reg_file, "r") as f:
            reg_data = json.load(f)

        set_b = reg_data["set_b_pm25_history"]["features"]
        set_domain = reg_data["domain_reduced"]["features"]

        assert len(set_b) == 29, f"Expected 29 features for set_b_pm25_history, got {len(set_b)}"
        assert len(set_domain) == 15, f"Expected 15 features for domain_reduced, got {len(set_domain)}"

        # 2. Exploratory Third Feature Set: set_b_plus_core_environment (29 PM2.5 history + 9 core env = 38)
        core_env_candidates = [
            "wind_speed_kmh_roll_mean_30d",
            "wind_x",
            "wind_y",
            "temperature_c_roll_mean_3d",
            "humidity_pct_roll_mean_3d",
            "fire_hotspot_sum_7d",
            "wind_weighted_hotspot_transport_score",
            "is_stubble_season",
            "is_monsoon"
        ]

        # Verify all candidates are prediction-safe
        valid_core_env = [c for c in core_env_candidates if c in safe_cols]
        set_b_plus_env = sorted(list(set(set_b).union(set(valid_core_env))))

        feature_sets = {
            "set_b_pm25_history": sorted(set_b),
            "domain_reduced": sorted(set_domain),
            "set_b_plus_core_environment": set_b_plus_env
        }

        for set_name, f_list in feature_sets.items():
            for feat in f_list:
                assert feat in safe_cols, f"Unsafe feature '{feat}' found in '{set_name}'!"
                assert feat != "date" and feat != "pm25"

        logger.info(f"Loaded Phase 3D feature sets: set_b_pm25_history ({len(set_b)}), domain_reduced ({len(set_domain)}), set_b_plus_core_environment ({len(set_b_plus_env)})")
        return feature_sets


if __name__ == "__main__":
    mgr = FeatureSetManager()
    fsets = mgr.get_phase3d_feature_sets()
    for name, cols in fsets.items():
        print(f"{name}: {len(cols)} features")
