import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import shap

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.cache import CachePhase4E
from ml.src.modeling.phase4e.response_schema import (
    AttributionResponse,
    FeatureAttributionItem,
    GroupAttributionItem,
    SCIENTIFIC_DISCLAIMER,
    PERSISTENCE_CAVEAT
)


class SHAPServicePhase4E:
    """
    AtmosIQ Phase 4E TreeSHAP Attribution Service.
    Computes exact TreeSHAP feature and group attributions for any requested date.
    """

    def __init__(self, data_loader: DataLoaderPhase4E = None, cache: CachePhase4E = None):
        self.loader = data_loader or DataLoaderPhase4E()
        self.cache = cache or CachePhase4E(self.loader)
        self.explainer = shap.TreeExplainer(self.loader.model)
        self.base_value = float(self.explainer.expected_value[0]) if isinstance(self.explainer.expected_value, (list, np.ndarray)) else float(self.explainer.expected_value)

        # Mapping feature_name -> attribution_group
        self.feature_group_map = dict(zip(self.loader.attr_groups_df["feature_name"], self.loader.attr_groups_df["attribution_group"]))

    def explain_prediction(self, date_str: str) -> AttributionResponse:
        """Serves TreeSHAP feature and group explanations for a date."""
        if date_str not in self.cache.date_to_index:
            raise ValueError(f"DATE_NOT_FOUND: Date '{date_str}' not found in Dataset v2.")

        idx = self.cache.date_to_index[date_str]
        x_row = self.loader.X_v2.iloc[[idx]]
        pred_val = float(self.loader.model.predict(x_row)[0])

        shap_vals = self.explainer.shap_values(x_row)[0]

        # Build feature list
        feat_items = []
        group_sums = {
            "pm25_persistence": 0.0,
            "meteorology": 0.0,
            "wind_ventilation": 0.0,
            "biomass_burning": 0.0,
            "calendar_seasonal": 0.0
        }
        group_abs_sums = {
            "pm25_persistence": 0.0,
            "meteorology": 0.0,
            "wind_ventilation": 0.0,
            "biomass_burning": 0.0,
            "calendar_seasonal": 0.0
        }

        for fname, s_val in zip(self.loader.feature_names, shap_vals):
            grp = self.feature_group_map.get(fname, "meteorology")
            val = float(s_val)
            feat_items.append(FeatureAttributionItem(
                feature_name=fname,
                shap_value=round(val, 4),
                attribution_group=grp
            ))

            if grp in group_sums:
                group_sums[grp] += val
                group_abs_sums[grp] += abs(val)
            else:
                group_sums["meteorology"] += val
                group_abs_sums["meteorology"] += abs(val)

        # Sort features by positive / negative
        pos_features = sorted([f for f in feat_items if f.shap_value > 0], key=lambda x: x.shap_value, reverse=True)[:5]
        neg_features = sorted([f for f in feat_items if f.shap_value < 0], key=lambda x: x.shap_value)[:5]

        total_abs_shap = sum(group_abs_sums.values()) or 1.0

        group_items = []
        for grp in ["pm25_persistence", "biomass_burning", "wind_ventilation", "meteorology", "calendar_seasonal"]:
            signed_sum = group_sums[grp]
            abs_sum = group_abs_sums[grp]
            share = (abs_sum / total_abs_shap) * 100.0
            group_items.append(GroupAttributionItem(
                group_name=grp,
                signed_shap_sum=round(signed_sum, 2),
                abs_shap_sum=round(abs_sum, 2),
                share_pct=round(share, 1)
            ))

        dominant = max(group_items, key=lambda g: g.abs_shap_sum).group_name

        return AttributionResponse(
            date=date_str,
            predicted_pm25=round(pred_val, 2),
            base_value=round(self.base_value, 2),
            top_positive_features=pos_features,
            top_negative_features=neg_features,
            group_attributions=group_items,
            dominant_group=dominant,
            scientific_disclaimer=SCIENTIFIC_DISCLAIMER,
            persistence_caveat=PERSISTENCE_CAVEAT
        )


if __name__ == "__main__":
    service = SHAPServicePhase4E()
    res = service.explain_prediction("2024-11-16")
    print(res.model_dump_json(indent=2))
