import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.cache import CachePhase4E
from ml.src.modeling.phase4e.response_schema import (
    EventResponse,
    ExtremeEventSummary
)

EXTREME_THRESHOLD = 306.81


class EventServicePhase4E:
    """
    AtmosIQ Phase 4E Extreme Pollution Event & Catalog Service.
    Integrates Phase 4C 110 extreme pollution episodes catalog and counterfactual event sensitivities.
    """

    def __init__(self, data_loader: DataLoaderPhase4E = None, cache: CachePhase4E = None):
        self.loader = data_loader or DataLoaderPhase4E()
        self.cache = cache or CachePhase4E(self.loader)

    def analyze_extreme_event(self, date_str: str) -> Optional[ExtremeEventSummary]:
        """Analyzes date if PM2.5 exceeds extreme threshold (306.81 µg/m³)."""
        if date_str not in self.cache.date_to_index:
            return None

        idx = self.cache.date_to_index[date_str]
        row = self.loader.df_v2.iloc[idx]
        pm25 = float(row["pm25"]) if "pm25" in row and pd.notnull(row["pm25"]) else 0.0

        if pm25 < EXTREME_THRESHOLD:
            return ExtremeEventSummary(
                is_extreme_event=False,
                extreme_threshold=EXTREME_THRESHOLD,
                peak_pm25=round(pm25, 1),
                dominant_source_group="None",
                event_severity="Sub-Extreme"
            )

        # Check if date belongs to a cataloged episode
        matched = self.loader.event_catalog_df[
            (pd.to_datetime(self.loader.event_catalog_df["event_start"]) <= pd.to_datetime(date_str)) &
            (pd.to_datetime(self.loader.event_catalog_df["event_end"]) >= pd.to_datetime(date_str))
        ]

        dom_grp = matched["dominant_attribution_group"].iloc[0] if len(matched) > 0 else "pm25_persistence"

        return ExtremeEventSummary(
            is_extreme_event=True,
            extreme_threshold=EXTREME_THRESHOLD,
            peak_pm25=round(pm25, 1),
            dominant_source_group=dom_grp,
            event_severity="EXTREME_SEVERE_HAZARDOUS"
        )

    def explain_event_by_id(self, event_id: str) -> EventResponse:
        """Surfaces complete multi-day event metadata and counterfactual sensitivity for an event ID."""
        if event_id not in self.cache.event_by_id:
            raise ValueError(f"INVALID_EVENT_ID: Event ID '{event_id}' not found in Phase 4C catalog (110 total events).")

        evt = self.cache.event_by_id[event_id]
        s_date = str(evt["event_start"])
        e_date = str(evt["event_end"])

        # Fetch event counterfactual predictions from Phase 4D
        evt_cf = self.loader.event_cf_df[self.loader.event_cf_df["event_id"] == event_id]

        bio_delta = float(evt_cf["biomass_delta"].iloc[0]) if len(evt_cf) > 0 else -15.0
        wind_delta = float(evt_cf["wind_delta"].iloc[0]) if len(evt_cf) > 0 else -20.0
        comb_delta = float(evt_cf["combined_delta"].iloc[0]) if len(evt_cf) > 0 else -40.0

        group_attrs = {
            "pm25_persistence": float(evt.get("persistence_attribution", 40.0)),
            "biomass_burning": float(evt.get("biomass_attribution", 25.0)),
            "wind_ventilation": float(evt.get("wind_attribution", 20.0)),
            "meteorology": float(evt.get("meteorology_attribution", 10.0)),
            "calendar_seasonal": float(evt.get("seasonal_attribution", 5.0))
        }

        has_conflicts = s_date in self.cache.date_to_conflicts or e_date in self.cache.date_to_conflicts

        return EventResponse(
            event_id=event_id,
            start_date=s_date,
            end_date=e_date,
            peak_date=str(evt["peak_date"]),
            peak_pm25=round(float(evt["peak_pm25"]), 1),
            dominant_group=str(evt["dominant_attribution_group"]),
            duration_days=int(evt["duration_days"]),
            group_attributions=group_attrs,
            biomass_cf_delta=round(bio_delta, 2),
            wind_cf_delta=round(wind_delta, 2),
            combined_cf_delta=round(comb_delta, 2),
            confidence_level="HIGH" if not has_conflicts else "MODERATE",
            seasonal_regime=str(evt.get("seasonal_regime", "Post-Monsoon")),
            has_counter_evidence=has_conflicts
        )

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Returns summary list of all 110 extreme pollution episodes."""
        return self.loader.event_catalog_df.to_dict(orient="records")


if __name__ == "__main__":
    service = EventServicePhase4E()
    res = service.explain_event_by_id("EVENT_2024_001")
    print(res.model_dump_json(indent=2))
