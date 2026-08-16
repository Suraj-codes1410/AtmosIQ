"""
AtmosIQ Phase 7B: State Data Structures for Atmospheric Trajectories.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class AtmosphericState:
    step_idx: int
    synthetic_date: str
    season: str
    pollution_regime: str

    # Core state variables
    pm25: float
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    wind_u_component_1d: float
    wind_v_component_1d: float
    pblh_1d: float
    pblh_min_1d: float
    rainfall_1d: float
    rain_event_1d: int
    ventilation_index_1d: float
    aod_550_1d: float
    fire_hotspot_count_1d: float
    upwind_stubble_quadrant_1d: float
    is_stubble_season: int
    festival_window: int

    # Constraint flags
    was_corrected: bool = False
    correction_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryBatch:
    trajectory_id: str
    length: int
    season: str
    target_regime: str
    states: List[AtmosphericState]

    def to_dataframe(self) -> pd.DataFrame:
        records = [s.to_dict() for s in self.states]
        df = pd.DataFrame(records)
        df["trajectory_id"] = self.trajectory_id
        return df
