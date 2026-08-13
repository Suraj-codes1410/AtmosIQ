import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("AttributionGrouperPhase4A")


class AttributionGrouperPhase4A:
    """
    AtmosIQ Phase 4A Feature Registry & Attribution Group Mapper.
    Constructs deterministic mapping from model features to environmental process attribution groups while preserving exact model feature ordering.
    """

    def __init__(self, package_dir: str = "ml/models/attribution/v1"):
        self.pkg_dir = Path(package_dir)
        self.pkg_dir.mkdir(parents=True, exist_ok=True)

        self.availability_file = Path("ml/data/modeling/v2/feature_availability.csv")
        assert self.availability_file.exists(), f"Feature availability file missing: {self.availability_file}"
        self.avail_df = pd.read_csv(self.availability_file)

        self.phase3g_feature_file = Path("ml/models/phase3g/feature_list.json")
        assert self.phase3g_feature_file.exists(), f"Phase 3G feature list missing: {self.phase3g_feature_file}"
        with open(self.phase3g_feature_file, "r") as f:
            self.model_features = json.load(f)["features"]

    @staticmethod
    def classify_attribution_group(f_name: str) -> tuple[str, str, str]:
        """Classifies a feature into an attribution group and provides mapping rationale and confidence."""
        if f_name.startswith("pm25_"):
            return "pm25_persistence", "Historical PM2.5 persistence and short-term rolling trajectory", "high"
        elif f_name.startswith("fire_hotspot_count_") or f_name == "is_stubble_season":
            return "biomass_burning", "Satellite MODIS/VIIRS upwind fire counts and stubble season indicator", "high"
        elif f_name.startswith("wind_speed_kmh_"):
            return "wind_ventilation", "Surface wind speed and atmospheric ventilation dispersion capability", "high"
        elif f_name.startswith("temperature_c_"):
            return "meteorology", "Ambient temperature and planetary boundary layer inversion dynamics", "high"
        elif f_name.startswith("humidity_pct_"):
            return "meteorology", "Relative humidity and secondary inorganic aerosol hydro-swelling", "high"
        elif f_name in ["festival_window"]:
            return "calendar_seasonal", "Cultural festival and emissions spike window", "high"
        else:
            return "unmapped", "Feature does not confidently match primary environmental process rules", "low"

    def build_registries(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Builds feature_registry.csv and attribution_groups.csv in exact model feature order."""
        logger.info("Constructing Phase 4A Feature Registry & Attribution Group Mappings...")

        avail_map = self.avail_df.set_index("feature_name").to_dict(orient="index")

        feat_registry_rows = []
        attr_group_rows = []

        for idx, f_name in enumerate(self.model_features, start=1):
            attr_grp, reason, confidence = self.classify_attribution_group(f_name)

            f_info = avail_map.get(f_name, {})

            feat_registry_rows.append({
                "feature_name": f_name,
                "feature_group": f_info.get("feature_group", "unknown"),
                "source_variable": f_info.get("source_variable", f_name.split("_")[0]),
                "feature_type": f_info.get("feature_type", "continuous"),
                "units": f_info.get("units", "various"),
                "temporal_availability": f_info.get("temporal_availability", "lagged"),
                "prediction_safe": f_info.get("prediction_safe", True),
                "model_feature_order": idx,
                "attribution_group": attr_grp
            })

            attr_group_rows.append({
                "feature_name": f_name,
                "attribution_group": attr_grp,
                "mapping_reason": reason,
                "confidence": confidence,
                "source_registry": "phase3g_production_feature_list"
            })

        feat_registry_df = pd.DataFrame(feat_registry_rows)
        attr_group_df = pd.DataFrame(attr_group_rows)

        feat_registry_df.to_csv(self.pkg_dir / "feature_registry.csv", index=False)
        attr_group_df.to_csv(self.pkg_dir / "attribution_groups.csv", index=False)

        logger.info(f"Registries created under {self.pkg_dir}. Total features: {len(feat_registry_df)}. Unmapped features: {(attr_group_df['attribution_group'] == 'unmapped').sum()}.")

        return feat_registry_df, attr_group_df


if __name__ == "__main__":
    grouper = AttributionGrouperPhase4A()
    grouper.build_registries()
