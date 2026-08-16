"""
AtmosIQ Phase 8A: Mandatory Extreme-Tail Environmental Filtering Engine.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class RejectionRecord:
    trajectory_id: str
    synthetic_date: str
    rejection_reason: str
    pm25: float
    ventilation_index: float
    precipitation: float
    season: str
    generator_seed: int
    generator_version: str
    phase_version: str


class ExtremeTailFilter:
    """
    Implements Phase 7C Mandatory Restriction C:
    Rejects synthetic extreme episodes where:
        PM2.5 >= 250.0 µg/m³
        AND
        (Ventilation Index > 4,500 m²/s OR Precipitation > 2.0 mm)
    """

    def __init__(
        self,
        enabled: bool = True,
        extreme_pm25_threshold: float = 250.0,
        vi_threshold: float = 4500.0,
        precipitation_threshold: float = 2.0,
    ):
        self.enabled = enabled
        self.extreme_pm25_threshold = extreme_pm25_threshold
        self.vi_threshold = vi_threshold
        self.precipitation_threshold = precipitation_threshold
        self.rejection_records: List[RejectionRecord] = []

    def evaluate_trajectory(
        self,
        df_traj: pd.DataFrame,
        trajectory_id: str,
        generator_seed: int,
        generator_version: str,
        phase_version: str = "Phase 8A v1.0.0"
    ) -> Tuple[bool, List[RejectionRecord]]:
        """
        Evaluates a trajectory for environmental extreme-tail coherence.
        Returns (is_accepted, list_of_rejections).
        """
        if not self.enabled:
            return True, []

        local_rejections = []

        # Find severe extreme rows
        extreme_mask = (df_traj["pm25"] >= self.extreme_pm25_threshold)
        if extreme_mask.any():
            for _, row in df_traj[extreme_mask].iterrows():
                vi_val = float(row["ventilation_index_1d"]) if "ventilation_index_1d" in row else 0.0
                rain_val = float(row["rainfall_1d"]) if "rainfall_1d" in row else 0.0

                vi_incoherent = (vi_val > self.vi_threshold)
                rain_incoherent = (rain_val > self.precipitation_threshold)

                if vi_incoherent or rain_incoherent:
                    reason_parts = []
                    if vi_incoherent:
                        reason_parts.append(f"VI ({vi_val:.1f} > {self.vi_threshold:.0f})")
                    if rain_incoherent:
                        reason_parts.append(f"Rain ({rain_val:.2f} > {self.precipitation_threshold:.1f})")

                    rec = RejectionRecord(
                        trajectory_id=trajectory_id,
                        synthetic_date=str(row.get("synthetic_date", "unknown")),
                        rejection_reason=f"EXTREME_TAIL_PHYSICAL_ENVIRONMENT_INCONSISTENCY: {', '.join(reason_parts)}",
                        pm25=float(row["pm25"]),
                        ventilation_index=vi_val,
                        precipitation=rain_val,
                        season=str(row.get("season", "unknown")),
                        generator_seed=generator_seed,
                        generator_version=generator_version,
                        phase_version=phase_version,
                    )
                    local_rejections.append(rec)
                    self.rejection_records.append(rec)

        is_accepted = (len(local_rejections) == 0)
        return is_accepted, local_rejections

    def get_rejection_dataframe(self) -> pd.DataFrame:
        """Returns dataframe of all recorded rejections."""
        if not self.rejection_records:
            return pd.DataFrame(columns=[
                "trajectory_id", "synthetic_date", "rejection_reason",
                "pm25", "ventilation_index", "precipitation",
                "season", "generator_seed", "generator_version", "phase_version"
            ])
        return pd.DataFrame([asdict(r) for r in self.rejection_records])
