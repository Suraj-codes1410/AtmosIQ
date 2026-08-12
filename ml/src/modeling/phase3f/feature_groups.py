import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureGroupsPhase3F")


class FeatureGroupManagerPhase3F:
    """
    AtmosIQ Phase 3F Feature Group Registry & Manager.
    Constructs canonical feature groups and targeted ablation sets strictly from Dataset v2 feature availability registry.
    """

    def __init__(self, availability_file: str = "ml/data/modeling/v2/feature_availability.csv"):
        self.avail_file = Path(availability_file)
        assert self.avail_file.exists(), f"Feature availability file missing: {self.avail_file}"
        self.avail_df = pd.read_csv(self.avail_file)
        self.safe_cols = self.avail_df[self.avail_df["prediction_safe"] == True]["feature_name"].tolist()

        # Load Phase 3C set_b_pm25_history definition (29 features)
        reg_file = Path("ml/experiments/phase3c/feature_set_registry.json")

        if reg_file.exists():
            with open(reg_file, "r") as f:
                reg_data = json.load(f)
            self.pm25_history_cols = [c for c in reg_data["set_b_pm25_history"]["features"] if c in self.safe_cols]
        else:
            self.pm25_history_cols = [c for c in self.safe_cols if "pm25" in c and c != "pm25"]

    def build_feature_groups(self) -> dict[str, list[str]]:
        """Constructs Groups A through G and exports feature_group_registry.csv."""
        logger.info("Constructing Phase 3F Feature Groups from Dataset v2 Registry...")

        # Categorize safe features by environmental domain
        weather_cols = [c for c in self.safe_cols if any(k in c for k in ["temperature", "temp", "humidity", "pressure", "wind", "precipitation", "precip", "thi", "rain"])]
        pollutant_cols = [c for c in self.safe_cols if any(k in c for k in ["pm10", "no2", "so2", "co", "o3"]) and "pm25" not in c]
        fire_cols = [c for c in self.safe_cols if any(k in c for k in ["fire", "hotspot", "brightness", "stubble", "punjab", "haryana", "rajasthan", "delhi_ncr_fire"])]
        transport_cols = [c for c in self.safe_cols if any(k in c for k in ["transport", "distance_weighted", "wind_weighted", "alignment", "ventilation"])]

        # Define Groups A through G
        group_a = ["pm25_lag_1d"] if "pm25_lag_1d" in self.safe_cols else [self.pm25_history_cols[0]]
        group_b = list(dict.fromkeys(self.pm25_history_cols))
        group_c = list(dict.fromkeys(group_b + weather_cols))
        group_d = list(dict.fromkeys(group_c + pollutant_cols))
        group_e = list(dict.fromkeys(group_c + fire_cols))
        group_f = list(dict.fromkeys(group_e + transport_cols))
        group_g = list(dict.fromkeys(self.safe_cols))

        groups = {
            "group_a_persistence": group_a,
            "group_b_pm25_history": group_b,
            "group_c_pm25_meteorology": group_c,
            "group_d_pm25_met_pollutants": group_d,
            "group_e_pm25_met_fire": group_e,
            "group_f_pm25_met_fire_transport": group_f,
            "group_g_full_safe": group_g,
            # Targeted Ablation Groups
            "ablation_pm25_history_only": group_b,
            "ablation_pm25_plus_weather": group_c,
            "ablation_pm25_plus_fire": list(dict.fromkeys(group_b + fire_cols)),
            "ablation_pm25_plus_fire_transport": list(dict.fromkeys(group_b + fire_cols + transport_cols)),
            "ablation_pm25_plus_pollutants": list(dict.fromkeys(group_b + pollutant_cols))
        }

        # Export feature_group_registry.csv
        exp_dir = Path("ml/experiments/phase3f")
        exp_dir.mkdir(parents=True, exist_ok=True)
        reg_rows = []
        for g_name, cols in groups.items():
            reg_rows.append({
                "group_id": g_name,
                "feature_count": len(cols),
                "feature_list": json.dumps(cols)
            })
        pd.DataFrame(reg_rows).to_csv(exp_dir / "feature_group_registry.csv", index=False)

        logger.info(f"Loaded {len(groups)} feature groups for Phase 3F. Group B: {len(group_b)}, Group G: {len(group_g)}.")
        return groups


if __name__ == "__main__":
    mgr = FeatureGroupManagerPhase3F()
    groups = mgr.build_feature_groups()
    for name, cols in groups.items():
        print(f"{name}: {len(cols)} features")
