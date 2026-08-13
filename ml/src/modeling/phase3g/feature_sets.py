import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureSetsPhase3G")


class FeatureSetManagerPhase3G:
    """
    AtmosIQ Phase 3G Feature Set Manager.
    Manages the 5 candidate feature sets for controlled hyperparameter tuning and model selection.
    """

    def __init__(self, availability_file: str = "ml/data/modeling/v2/feature_availability.csv"):
        self.avail_file = Path(availability_file)
        assert self.avail_file.exists(), f"Feature availability file missing: {self.avail_file}"
        self.avail_df = pd.read_csv(self.avail_file)
        self.safe_cols = set(self.avail_df[self.avail_df["prediction_safe"] == True]["feature_name"])

    def get_phase3g_feature_sets(self) -> dict[str, list[str]]:
        """Returns candidate feature sets for Phase 3G tuning."""
        # 1. Load set_b_pm25_history and domain_reduced from Phase 3C registry
        reg_file = Path("ml/experiments/phase3c/feature_set_registry.json")
        assert reg_file.exists(), f"Phase 3C registry missing: {reg_file}"

        with open(reg_file, "r") as f:
            reg_data = json.load(f)

        set_b = [c for c in reg_data["set_b_pm25_history"]["features"] if c in self.safe_cols]
        dom_red = [c for c in reg_data["domain_reduced"]["features"] if c in self.safe_cols]

        # 2. Build Environmental Feature Groups from Phase 3F logic
        weather_cols = [c for c in self.safe_cols if any(k in c for k in ["temperature", "temp", "humidity", "pressure", "wind", "precipitation", "precip", "thi", "rain"])]
        fire_cols = [c for c in self.safe_cols if any(k in c for k in ["fire", "hotspot", "brightness", "stubble", "punjab", "haryana", "rajasthan", "delhi_ncr_fire"])]
        transport_cols = [c for c in self.safe_cols if any(k in c for k in ["transport", "distance_weighted", "wind_weighted", "alignment", "ventilation"])]

        group_c = list(dict.fromkeys(set_b + weather_cols))
        group_e = list(dict.fromkeys(group_c + fire_cols))
        group_f = list(dict.fromkeys(group_e + transport_cols))

        feature_sets = {
            "set_b_pm25_history": set_b,
            "group_c_pm25_meteorology": group_c,
            "group_e_pm25_met_fire": group_e,
            "group_f_pm25_met_fire_transport": group_f,
            "domain_reduced": dom_red
        }

        logger.info(f"Loaded {len(feature_sets)} candidate feature sets for Phase 3G.")
        for name, cols in feature_sets.items():
            logger.info(f"  - {name}: {len(cols)} features")

        return feature_sets


if __name__ == "__main__":
    mgr = FeatureSetManagerPhase3G()
    fsets = mgr.get_phase3g_feature_sets()
