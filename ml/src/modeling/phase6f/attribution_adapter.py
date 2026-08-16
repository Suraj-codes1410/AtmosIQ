import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import shap
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6f.config import DecisionSupportConfigPhase6F

logger = setup_logger("AttributionAdapterPhase6F")


class AttributionAdapterPhase6F:
    """
    Production Attribution Adapter for Phase 6F.
    Integrates TreeSHAP explainability, sign stability metadata, and 6 environmental process groups.
    """

    GROUP_DEFINITIONS = {
        "pm25_persistence": [
            "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
            "pm25_roll_mean_3d", "pm25_roll_mean_7d", "pm25_roll_mean_14d",
            "pm25_roll_std_7d", "pm25_roll_max_7d", "pm25_roll_min_7d"
        ],
        "meteorology": [
            "temperature_c_lag_1d", "temperature_c_roll_mean_3d", "temperature_c_roll_min_3d",
            "humidity_pct_lag_1d", "humidity_pct_roll_mean_3d", "humidity_pct_roll_max_7d"
        ],
        "wind_ventilation": [
            "wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d",
            "wind_u_component_1d", "wind_v_component_1d",
            "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"
        ],
        "biomass_burning": [
            "is_stubble_season", "fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d",
            "fire_hotspot_count_roll_mean_7d", "upwind_stubble_quadrant_1d"
        ],
        "external_environmental": [
            "rainfall_1d", "rainfall_3d", "rain_event_1d", "washout_index_3d", "aod_550_1d"
        ],
        "calendar_seasonal": [
            "festival_window"
        ]
    }

    def __init__(self, config: DecisionSupportConfigPhase6F, model_path: str = "ml/models/production/v3/model.joblib", stability_path: str = "ml/experiments/phase6e/shap_feature_summary.csv"):
        self.config = config
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Load feature stability lookup from Phase 6E
        self.stability_lookup = {}
        if Path(stability_path).exists():
            df_stab = pd.read_csv(stability_path)
            for _, r in df_stab.iterrows():
                self.stability_lookup[r['feature_name']] = {
                    "stability_classification": r['stability_classification'],
                    "mean_sign_stability": float(r['mean_sign_stability']),
                    "mean_shap_std": float(r['mean_shap_std'])
                }

    def compute_attribution(self, x_vec: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Computes TreeSHAP attributions and environmental group breakdown for a single input vector.
        """
        if x_vec.ndim == 1:
            X_mat = x_vec.reshape(1, -1)
        else:
            X_mat = x_vec

        shap_values = self.explainer.shap_values(X_mat, check_additivity=False)[0]
        base_val = float(self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value)

        # Feature level details
        feature_contributions = []
        for j, feat_name in enumerate(feature_names):
            s_val = float(shap_values[j])
            stab_info = self.stability_lookup.get(feat_name, {
                "stability_classification": "MODERATE_STABILITY",
                "mean_sign_stability": 0.80,
                "mean_shap_std": 2.0
            })

            feature_contributions.append({
                "feature_name": feat_name,
                "feature_value": float(X_mat[0, j]),
                "shap_attribution": s_val,
                "absolute_shap": abs(s_val),
                "sign_stability": stab_info["mean_sign_stability"],
                "stability_category": stab_info["stability_classification"],
                "attribution_uncertainty_std": stab_info["mean_shap_std"]
            })

        # Sort features by absolute contribution
        feature_contributions.sort(key=lambda x: x["absolute_shap"], reverse=True)
        dominant_features = [f["feature_name"] for f in feature_contributions[:3]]

        # Group level aggregation
        feat_to_idx = {name: i for i, name in enumerate(feature_names)}
        group_contributions = []

        for grp_name, f_list in self.GROUP_DEFINITIONS.items():
            grp_indices = [feat_to_idx[f] for f in f_list if f in feat_to_idx]
            grp_shap = float(np.sum([shap_values[i] for i in grp_indices])) if grp_indices else 0.0
            
            group_contributions.append({
                "group_name": grp_name,
                "group_shap": grp_shap,
                "absolute_group_shap": abs(grp_shap),
                "feature_count": len(grp_indices)
            })

        group_contributions.sort(key=lambda x: x["absolute_group_shap"], reverse=True)
        dominant_groups = [g["group_name"] for g in group_contributions[:2]]

        return {
            "base_value": base_val,
            "feature_contributions": feature_contributions,
            "group_contributions": group_contributions,
            "dominant_features": dominant_features,
            "dominant_groups": dominant_groups
        }
