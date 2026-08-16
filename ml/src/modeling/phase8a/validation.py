"""
AtmosIQ Phase 8A: Physics Validation Engine.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Phase8APhysicsValidator:
    """Validates that all generated synthetic trajectories adhere 100% to physical laws."""

    def __init__(self):
        pass

    def validate_trajectory(self, df_traj: pd.DataFrame, trajectory_id: str) -> Tuple[bool, Dict[str, Any]]:
        violations = []

        # 1. PM2.5 Non-Negativity
        neg_pm = int((df_traj["pm25"] < 0.0).sum())
        if neg_pm > 0:
            violations.append(f"PM2.5 < 0.0 ({neg_pm} rows)")

        # 2. Wind Speed Non-Negativity
        neg_ws = int((df_traj["wind_speed_kmh"] < 0.0).sum())
        if neg_ws > 0:
            violations.append(f"Wind speed < 0.0 ({neg_ws} rows)")

        # 3. Rainfall Non-Negativity
        neg_rain = int((df_traj["rainfall_1d"] < 0.0).sum())
        if neg_rain > 0:
            violations.append(f"Rainfall < 0.0 ({neg_rain} rows)")

        # 4. PBLH Positive Bounds (>= 150m)
        bad_pblh = int((df_traj["pblh_1d"] < 150.0).sum())
        if bad_pblh > 0:
            violations.append(f"PBLH < 150.0m ({bad_pblh} rows)")

        # 5. Humidity Bounds (0 - 100%)
        bad_rh = int(((df_traj["humidity_pct"] < 0.0) | (df_traj["humidity_pct"] > 100.0)).sum())
        if bad_rh > 0:
            violations.append(f"Humidity outside [0, 100]% ({bad_rh} rows)")

        # 6. Temperature Bounds (0 - 50°C)
        bad_temp = int(((df_traj["temperature_c"] < 0.0) | (df_traj["temperature_c"] > 50.0)).sum())
        if bad_temp > 0:
            violations.append(f"Temperature outside [0, 50]C ({bad_temp} rows)")

        # 7. Fire Counts Non-Negativity
        neg_fire = int((df_traj["fire_hotspot_count_1d"] < 0.0).sum())
        if neg_fire > 0:
            violations.append(f"Fire hotspot count < 0 ({neg_fire} rows)")

        # 8. Ventilation Index Hydrodynamic Identity (VI = ws_ms * PBLH)
        ws_ms = df_traj["wind_speed_kmh"] * (1000.0 / 3600.0)
        expected_vi = ws_ms * df_traj["pblh_1d"]
        vi_diff = np.abs(df_traj["ventilation_index_1d"] - expected_vi)
        bad_vi = int((vi_diff > 1.0).sum())
        if bad_vi > 0:
            violations.append(f"Hydrodynamic identity mismatch in VI ({bad_vi} rows)")

        # 9. Rain Event Binary Logic
        bad_rain_event = int(((df_traj["rain_event_1d"] == 1) != (df_traj["rainfall_1d"] >= 1.0)).sum())
        if bad_rain_event > 0:
            violations.append(f"Rain event indicator logic mismatch ({bad_rain_event} rows)")

        # 10. NaN / Inf Prevention
        nan_count = int(df_traj.isna().sum().sum())
        if nan_count > 0:
            violations.append(f"Contains {nan_count} NaN values")
        inf_count = int(np.isinf(df_traj.select_dtypes(include=[np.number]).values).sum())
        if inf_count > 0:
            violations.append(f"Contains {inf_count} infinite values")

        is_valid = (len(violations) == 0)

        report = {
            "trajectory_id": trajectory_id,
            "is_physically_valid": is_valid,
            "violation_count": len(violations),
            "violation_details": violations,
            "observation_count": len(df_traj),
        }

        return is_valid, report
