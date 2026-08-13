import sys
import json
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import shap
from ml.src.utils.logger import setup_logger

logger = setup_logger("SHAPEnginePhase4B")


class SHAPEnginePhase4B:
    """
    AtmosIQ Phase 4B TreeSHAP Engine.
    Loads frozen Phase 4A Attribution Package, verifies exact feature order, computes TreeSHAP values for all observations, and generates wide and long-form SHAP datasets.
    """

    def __init__(self, pkg_dir: str = "ml/models/attribution/v1", exp_dir: str = "ml/experiments/phase4b"):
        self.pkg_dir = Path(pkg_dir)
        self.exp_dir = Path(exp_dir)
        self.shap_dir = self.exp_dir / "shap_values"
        self.shap_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.pkg_dir / "model.joblib"
        self.feat_reg_path = self.pkg_dir / "feature_registry.csv"
        self.attr_group_path = self.pkg_dir / "attribution_groups.csv"

        assert self.model_path.exists(), f"Phase 4A model missing: {self.model_path}"
        assert self.feat_reg_path.exists(), f"Feature registry missing: {self.feat_reg_path}"
        assert self.attr_group_path.exists(), f"Attribution groups missing: {self.attr_group_path}"

        self.ds_file = Path("ml/data/modeling/v2/feature_dataset_frozen.csv")
        assert self.ds_file.exists(), f"Dataset v2 missing: {self.ds_file}"

        self.feat_reg = pd.read_csv(self.feat_reg_path).sort_values("model_feature_order")
        self.feature_order = self.feat_reg["feature_name"].tolist()

        self.attr_groups_df = pd.read_csv(self.attr_group_path)
        self.feature_to_group = self.attr_groups_df.set_index("feature_name")["attribution_group"].to_dict()

        self.model = joblib.load(self.model_path)
        self.df = pd.read_csv(self.ds_file)
        self.df["date_dt"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date_dt").reset_index(drop=True)

        # Verify X columns match feature order exactly
        self.X_all = self.df[self.feature_order]
        assert self.X_all.columns.tolist() == self.feature_order, "CRITICAL ERROR: Feature ordering does not match feature_registry.csv!"

    def compute_shap_values(self) -> dict:
        """Computes TreeSHAP values for all Dataset v2 observations."""
        logger.info(f"Instantiating TreeExplainer for frozen Random Forest model ({len(self.feature_order)} features)...")

        explainer = shap.TreeExplainer(self.model)

        # Handle base value (expected_value can be scalar or numpy array)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[0])
        else:
            base_value = float(explainer.expected_value)

        logger.info(f"TreeExplainer expected base value: {base_value:.4f} µg/m³.")
        logger.info(f"Computing TreeSHAP values for all {len(self.df)} daily observations...")

        shap_matrix = explainer.shap_values(self.X_all)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[0]

        predictions = self.model.predict(self.X_all)

        # Build wide SHAP DataFrame
        shap_col_names = [f"shap_{col}" for col in self.feature_order]
        wide_shap_df = pd.DataFrame(shap_matrix, columns=shap_col_names)
        wide_shap_df.insert(0, "base_value", base_value)
        wide_shap_df.insert(0, "predicted_pm25", predictions)
        wide_shap_df.insert(0, "actual_pm25", self.df["pm25"].values)
        wide_shap_df.insert(0, "date", self.df["date"].values)

        # Slice validation (2022-2023) and test (2024)
        dates_dt = self.df["date_dt"]
        val_mask = (dates_dt >= "2022-01-01") & (dates_dt <= "2023-12-31")
        test_mask = (dates_dt >= "2024-01-01") & (dates_dt <= "2024-12-31")

        wide_shap_df[test_mask].to_csv(self.shap_dir / "shap_values_test.csv", index=False)
        wide_shap_df[val_mask].to_csv(self.shap_dir / "shap_values_validation.csv", index=False)
        wide_shap_df.to_csv(self.shap_dir / "shap_values_all.csv", index=False)

        # Build long-form SHAP table
        logger.info("Generating long-form SHAP table (shap_values_long.csv)...")
        long_rows = []
        for i in range(len(self.df)):
            dt = self.df.loc[i, "date"]
            act = self.df.loc[i, "pm25"]
            pred = predictions[i]
            for j, f_name in enumerate(self.feature_order):
                f_val = self.X_all.iloc[i, j]
                s_val = shap_matrix[i, j]
                abs_s = abs(s_val)
                direction = "positive" if s_val > 1e-6 else ("negative" if s_val < -1e-6 else "zero")
                grp = self.feature_to_group.get(f_name, "unmapped")

                long_rows.append({
                    "date": dt,
                    "actual_pm25": act,
                    "predicted_pm25": pred,
                    "base_value": base_value,
                    "feature_name": f_name,
                    "feature_value": f_val,
                    "shap_value": s_val,
                    "attribution_group": grp,
                    "absolute_shap_value": abs_s,
                    "contribution_direction": direction
                })

        long_df = pd.DataFrame(long_rows)
        long_df.to_csv(self.exp_dir / "shap_values_long.csv", index=False)

        logger.info(f"TreeSHAP calculation complete. Wide and long SHAP tables exported under {self.exp_dir}.")

        return {
            "explainer": explainer,
            "base_value": base_value,
            "shap_matrix": shap_matrix,
            "predictions": predictions,
            "wide_df": wide_shap_df,
            "long_df": long_df,
            "X_all": self.X_all,
            "df": self.df
        }


if __name__ == "__main__":
    engine = SHAPEnginePhase4B()
    engine.compute_shap_values()
