import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("GroupAttributionPhase4B")


class GroupAttributionPhase4B:
    """
    AtmosIQ Phase 4B Group Attribution Aggregator.
    Aggregates feature-level TreeSHAP values into 5 environmental process attribution groups and calculates group reconstruction metrics.
    """

    def __init__(self, pkg_dir: str = "ml/models/attribution/v1", exp_dir: str = "ml/experiments/phase4b"):
        self.pkg_dir = Path(pkg_dir)
        self.exp_dir = Path(exp_dir)
        self.group_dir = self.exp_dir / "group_attributions"
        self.group_dir.mkdir(parents=True, exist_ok=True)

        self.attr_group_path = self.pkg_dir / "attribution_groups.csv"
        assert self.attr_group_path.exists(), f"Attribution groups missing: {self.attr_group_path}"
        self.attr_groups_df = pd.read_csv(self.attr_group_path)

        self.ordered_groups = [
            "pm25_persistence",
            "meteorology",
            "wind_ventilation",
            "biomass_burning",
            "calendar_seasonal"
        ]

    def aggregate_groups(self, df: pd.DataFrame, feature_order: list, shap_matrix: np.ndarray, base_value: float, predictions: np.ndarray = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Aggregates feature SHAP values by group for every observation."""
        logger.info("Aggregating feature-level SHAP values into 5 environmental process groups...")

        group_indices = {grp: [] for grp in self.ordered_groups}
        feature_to_group = self.attr_groups_df.set_index("feature_name")["attribution_group"].to_dict()

        for idx, f_name in enumerate(feature_order):
            grp = feature_to_group.get(f_name, "unmapped")
            if grp in group_indices:
                group_indices[grp].append(idx)
            else:
                logger.warning(f"Unmapped feature group '{grp}' found for feature '{f_name}'!")

        group_shap_cols = {}
        group_shap_matrix = np.zeros((len(df), len(self.ordered_groups)))

        for g_idx, grp in enumerate(self.ordered_groups):
            indices = group_indices[grp]
            if indices:
                g_vals = shap_matrix[:, indices].sum(axis=1)
            else:
                g_vals = np.zeros(len(df))

            group_shap_matrix[:, g_idx] = g_vals
            group_shap_cols[f"{grp}_shap"] = g_vals

        if predictions is None:
            predictions = df["predicted_pm25"].values if "predicted_pm25" in df.columns else df["pm25"].values

        reconstructed = base_value + group_shap_matrix.sum(axis=1)
        recon_error = predictions - reconstructed

        group_df = pd.DataFrame({
            "date": df["date"].values,
            "actual_pm25": df["actual_pm25"].values if "actual_pm25" in df.columns else df["pm25"].values,
            "predicted_pm25": predictions,
            "base_value": base_value,
            **group_shap_cols,
            "reconstruction_error": recon_error
        })

        # Save train, val, test slices
        dates_dt = pd.to_datetime(df["date"])
        val_mask = (dates_dt >= "2022-01-01") & (dates_dt <= "2023-12-31")
        test_mask = (dates_dt >= "2024-01-01") & (dates_dt <= "2024-12-31")

        group_df[test_mask].to_csv(self.group_dir / "group_attributions_test.csv", index=False)
        group_df[val_mask].to_csv(self.group_dir / "group_attributions_validation.csv", index=False)
        group_df.to_csv(self.group_dir / "group_attributions_all.csv", index=False)

        logger.info(f"Group attributions saved under {self.group_dir}. Max group reconstruction error: {np.abs(recon_error).max():.4e}.")

        return group_df, group_shap_matrix


if __name__ == "__main__":
    aggregator = GroupAttributionPhase4B()
