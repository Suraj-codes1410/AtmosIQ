import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E


class CachePhase4E:
    """
    In-memory indexed lookup cache for AtmosIQ Phase 4E.
    Provides sub-millisecond date and event index access.
    """

    def __init__(self, data_loader: DataLoaderPhase4E = None):
        self.loader = data_loader or DataLoaderPhase4E()
        self._build_indices()

    def _build_indices(self):
        # 1. Date to row index
        dates = self.loader.df_v2["date"].tolist()
        self.date_to_index = {d: i for i, d in enumerate(dates)}

        # 2. Date to Group SHAP
        self.date_to_group_shap = {}
        if not self.loader.group_shap_df.empty:
            for _, row in self.loader.group_shap_df.iterrows():
                d = str(row["date"])
                self.date_to_group_shap[d] = {
                    "pm25_persistence": float(row.get("pm25_persistence", 0.0)),
                    "meteorology": float(row.get("meteorology", 0.0)),
                    "wind_ventilation": float(row.get("wind_ventilation", 0.0)),
                    "biomass_burning": float(row.get("biomass_burning", 0.0)),
                    "calendar_seasonal": float(row.get("calendar_seasonal", 0.0))
                }

        # 3. Date to Conflicts
        self.date_to_conflicts = {}
        for _, row in self.loader.conflicts_df.iterrows():
            d = str(row["date"])
            grp = str(row.get("attribution_group", row.get("group", "unknown")))
            reason = str(row.get("conflict_type", row.get("conflict_reason", "Environmental Indicator Conflict")))
            if d not in self.date_to_conflicts:
                self.date_to_conflicts[d] = []
            self.date_to_conflicts[d].append({
                "group": grp,
                "reason": reason,
                "severity": str(row.get("severity", "moderate"))
            })

        # 4. Date to Counterfactuals
        self.date_to_cf = {}
        for _, row in self.loader.cf_results_df.iterrows():
            d = str(row["date"])
            s = str(row["scenario"])
            if d not in self.date_to_cf:
                self.date_to_cf[d] = {}
            self.date_to_cf[d][s] = {
                "date": d,
                "scenario": s,
                "target_group": str(row["target_group"]),
                "baseline_prediction": float(row["prediction_observed"]),
                "counterfactual_prediction": float(row["prediction_counterfactual"]),
                "delta_prediction": float(row["delta_prediction"])
            }

        # 5. Date to OOD flag
        self.date_to_ood = {}
        for _, row in self.loader.ood_df.iterrows():
            d = str(row["date"])
            self.date_to_ood[d] = bool(row.get("is_ood", False))

        # 6. Event catalog by ID
        self.event_by_id = {}
        for _, row in self.loader.event_catalog_df.iterrows():
            eid = str(row["event_id"])
            self.event_by_id[eid] = row.to_dict()


if __name__ == "__main__":
    cache = CachePhase4E()
    print(f"Indexed {len(cache.date_to_index)} dates in cache.")
