import sys
from pathlib import Path
from typing import List
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.cache import CachePhase4E
from ml.src.modeling.phase4e.response_schema import (
    EnvironmentalValidationResponse,
    GroupValidationEvidence,
    CounterEvidenceItem
)


class ValidationServicePhase4E:
    """
    AtmosIQ Phase 4E Environmental Validation & Counter-Evidence Service.
    Integrates Phase 4C independent environmental evidence and surfaces evidence conflicts.
    """

    def __init__(self, data_loader: DataLoaderPhase4E = None, cache: CachePhase4E = None):
        self.loader = data_loader or DataLoaderPhase4E()
        self.cache = cache or CachePhase4E(self.loader)

    def validate_attribution(self, date_str: str) -> EnvironmentalValidationResponse:
        """Surfaces independent environmental evidence and counter-evidence for a date."""
        if date_str not in self.cache.date_to_index:
            raise ValueError(f"DATE_NOT_FOUND: Date '{date_str}' not found in Dataset v2.")

        idx = self.cache.date_to_index[date_str]
        row = self.loader.df_v2.iloc[idx]

        # Extract independent indicators
        fire_count = float(row["fire_hotspot_count"]) if "fire_hotspot_count" in row and pd.notnull(row["fire_hotspot_count"]) else 0.0
        wind_spd = float(row["wind_speed_kmh"]) if "wind_speed_kmh" in row and pd.notnull(row["wind_speed_kmh"]) else 0.0
        temp = float(row["temperature_c"]) if "temperature_c" in row and pd.notnull(row["temperature_c"]) else 0.0
        pblh = float(row["pblh_m"]) if "pblh_m" in row and pd.notnull(row["pblh_m"]) else 0.0

        evidences = [
            GroupValidationEvidence(
                group_name="biomass_burning",
                supporting_indicator="satellite_fire_hotspots_modis_viirs",
                relationship="positive",
                evidence_status="PASS" if fire_count > 50 else ("NEUTRAL" if fire_count > 10 else "LOW_FIRE"),
                observed_value=round(fire_count, 1)
            ),
            GroupValidationEvidence(
                group_name="wind_ventilation",
                supporting_indicator="surface_wind_speed_kmh",
                relationship="inverse",
                evidence_status="STAGNANT_PASS" if wind_spd < 10.0 else "DISPERSION_PASS",
                observed_value=round(wind_spd, 1)
            ),
            GroupValidationEvidence(
                group_name="meteorology",
                supporting_indicator="boundary_layer_height_pblh_m",
                relationship="inverse",
                evidence_status="INVERSION_PASS" if pblh < 400.0 else "NORMAL_MET",
                observed_value=round(pblh, 1) if pblh > 0 else round(temp, 1)
            )
        ]

        # Check for explicit conflicts in Phase 4C conflicts CSV
        conflicts_raw = self.cache.date_to_conflicts.get(date_str, [])
        counter_items = [
            CounterEvidenceItem(
                group=c["group"],
                reason=c["reason"],
                severity=c.get("severity", "moderate")
            )
            for c in conflicts_raw
        ]

        has_conflicts = len(counter_items) > 0
        overall_status = "WARNING_CONFLICT" if has_conflicts else "PASS"

        return EnvironmentalValidationResponse(
            date=date_str,
            validation_status=overall_status,
            group_evidence=evidences,
            has_counter_evidence=has_conflicts,
            counter_evidence_conflicts=counter_items
        )


if __name__ == "__main__":
    service = ValidationServicePhase4E()
    res = service.validate_attribution("2024-02-01")
    print(res.model_dump_json(indent=2))
