import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureInterventionPhase4D")


class FeatureInterventionEnginePhase4D:
    """
    AtmosIQ Phase 4D Controlled Feature Intervention Engine.
    Modifies ONLY features belonging to the targeted attribution group(s) while leaving all other features 100% exact and unchanged.
    """

    def __init__(self, pkg_dir: str = "ml/models/attribution/v1"):
        self.pkg_dir = Path(pkg_dir)
        self.model_path = self.pkg_dir / "model.joblib"
        self.feat_reg_path = self.pkg_dir / "feature_registry.csv"
        self.attr_groups_path = self.pkg_dir / "attribution_groups.csv"

        assert self.model_path.exists(), f"Model missing at {self.model_path}!"
        assert self.feat_reg_path.exists(), f"Feature registry missing at {self.feat_reg_path}!"
        assert self.attr_groups_path.exists(), f"Attribution groups file missing at {self.attr_groups_path}!"

        self.model = joblib.load(self.model_path)
        feat_reg_df = pd.read_csv(self.feat_reg_path)
        attr_groups_df = pd.read_csv(self.attr_groups_path)

        if "model_feature_order" in feat_reg_df.columns:
            merged = pd.merge(feat_reg_df, attr_groups_df[["feature_name", "attribution_group"]], on="feature_name", how="left", suffixes=("", "_grp"))
            if "attribution_group_grp" in merged.columns:
                merged["attribution_group"] = merged["attribution_group_grp"].fillna(merged["attribution_group"])
            self.attr_df = merged.sort_values("model_feature_order")
        else:
            self.attr_df = attr_groups_df

        self.feature_order = self.attr_df["feature_name"].tolist()
        self.group_mapping = dict(zip(self.attr_df["feature_name"], self.attr_df["attribution_group"]))

        # Compute training population reference quantiles for each feature
        self.reference_quantiles = {}

    def fit_reference_quantiles(self, X_ref: pd.DataFrame):
        """Fits reference quantiles (Q10, Q25, Q50, Q75, Q90) from reference feature dataset."""
        for col in self.feature_order:
            vals = X_ref[col].values
            self.reference_quantiles[col] = {
                "min": float(np.min(vals)),
                "q10": float(np.percentile(vals, 10)),
                "q25": float(np.percentile(vals, 25)),
                "q50": float(np.percentile(vals, 50)),
                "q75": float(np.percentile(vals, 75)),
                "q90": float(np.percentile(vals, 90)),
                "max": float(np.max(vals))
            }
        logger.info("Reference quantiles calculated for all 147 prediction-safe features.")

    def apply_intervention(self, x_obs: np.ndarray, target_group: str, quantile_key: str = "q25") -> np.ndarray:
        """
        Creates counterfactual feature vector x_cf modifying ONLY features belonging to target_group.
        All untargeted features remain exactly identical to x_obs.
        """
        x_cf = x_obs.copy()
        for idx, feat_name in enumerate(self.feature_order):
            grp = self.group_mapping[feat_name]
            if grp == target_group:
                ref_val = self.reference_quantiles[feat_name][quantile_key]
                x_cf[idx] = ref_val
        return x_cf

    def apply_multi_group_intervention(self, x_obs: np.ndarray, group_quantile_map: dict) -> np.ndarray:
        """
        Creates counterfactual feature vector modifying multiple specified target groups.
        group_quantile_map example: {"biomass_burning": "q25", "wind_ventilation": "q75"}
        """
        x_cf = x_obs.copy()
        for idx, feat_name in enumerate(self.feature_order):
            grp = self.group_mapping[feat_name]
            if grp in group_quantile_map:
                q_key = group_quantile_map[grp]
                ref_val = self.reference_quantiles[feat_name][q_key]
                x_cf[idx] = ref_val
        return x_cf

    def predict_observed_and_counterfactual(self, x_obs: np.ndarray, x_cf: np.ndarray) -> tuple[float, float, float]:
        """
        Predicts baseline f(x_obs) and counterfactual f(x_cf).
        Returns (prediction_observed, prediction_counterfactual, delta_prediction).
        """
        df_obs = pd.DataFrame([x_obs], columns=self.feature_order)
        df_cf = pd.DataFrame([x_cf], columns=self.feature_order)
        pred_obs = float(self.model.predict(df_obs)[0])
        pred_cf = float(self.model.predict(df_cf)[0])
        delta = pred_cf - pred_obs
        return pred_obs, pred_cf, delta

    def predict_batch(self, X_mat: np.ndarray) -> np.ndarray:
        """Runs bulk predictions on 2D matrix with feature names."""
        df_batch = pd.DataFrame(X_mat, columns=self.feature_order)
        return self.model.predict(df_batch)


if __name__ == "__main__":
    engine = FeatureInterventionEnginePhase4D()
