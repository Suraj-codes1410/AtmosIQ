"""
AtmosIQ Phase 7B: Physics Constraint Engine.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


class PhysicsConstraintEnginePhase7B:
    """
    Enforces 10 hard physical constraints:
    1. PM2.5 non-negativity (PM2.5 >= 0)
    2. Boundary layer height positive (PBLH >= 150m)
    3. Precipitation non-negativity (Rainfall >= 0)
    4. Wind speed non-negativity (ws >= 0)
    5. Relative humidity within physical bounds (0 <= RH <= 100%)
    6. Temperature within climatological bounds (0 <= T <= 50°C)
    7. Active fire count non-negativity (Fires >= 0)
    8. Hydrodynamic ventilation index consistency (VI = ws * PBLH)
    9. Precipitation binary indicator consistency (rain_event = I(rain >= 1mm))
    10. Washout / extreme smog joint consistency (no PM2.5 > 300 with Rain > 30mm)
    """

    def __init__(self):
        self.audit_records: List[Dict[str, Any]] = []

    def evaluate_and_constrain(
        self,
        raw_state: Dict[str, Any],
        trajectory_id: str,
        step_idx: int
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Evaluates and applies bounded physical corrections if necessary.
        Returns: (constrained_state, status, notes)
        status in ["PASS", "CORRECTED", "REJECTED"]
        """
        state = dict(raw_state)
        was_corrected = False
        notes = []

        # 1. PM2.5 non-negativity
        if state["pm25"] < 0.0:
            notes.append(f"PM2.5 clipped from {state['pm25']:.2f} to 15.0")
            state["pm25"] = 15.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "pm25_non_negative", "CORRECTED", state["pm25"])
        elif state["pm25"] > 500.0:
            notes.append(f"PM2.5 clipped from {state['pm25']:.2f} to 450.0")
            state["pm25"] = 450.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "pm25_upper_bound", "CORRECTED", state["pm25"])

        # 2. PBLH bounds
        if state["pblh_1d"] < 150.0:
            notes.append(f"PBLH clipped from {state['pblh_1d']:.1f} to 200.0")
            state["pblh_1d"] = 200.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "pblh_min_bound", "CORRECTED", state["pblh_1d"])

        # 3. Rainfall bounds
        if state["rainfall_1d"] < 0.0:
            state["rainfall_1d"] = 0.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "rainfall_non_negative", "CORRECTED", 0.0)

        # 4. Wind speed bounds
        if state["wind_speed_kmh"] < 0.0:
            state["wind_speed_kmh"] = 2.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "wind_speed_non_negative", "CORRECTED", 2.0)

        # 5. Humidity bounds
        if state["humidity_pct"] < 5.0:
            state["humidity_pct"] = 10.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "humidity_lower_bound", "CORRECTED", 10.0)
        elif state["humidity_pct"] > 100.0:
            state["humidity_pct"] = 98.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "humidity_upper_bound", "CORRECTED", 98.0)

        # 6. Temperature bounds
        if state["temperature_c"] < 0.0:
            state["temperature_c"] = 5.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "temperature_lower_bound", "CORRECTED", 5.0)
        elif state["temperature_c"] > 50.0:
            state["temperature_c"] = 45.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "temperature_upper_bound", "CORRECTED", 45.0)

        # 7. Fire counts
        if state["fire_hotspot_count_1d"] < 0.0:
            state["fire_hotspot_count_1d"] = 0.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "fire_count_non_negative", "CORRECTED", 0.0)

        # 8. Ventilation index exact hydrodynamic consistency
        ws_ms = state["wind_speed_kmh"] * (1000.0 / 3600.0)
        exact_vi = ws_ms * state["pblh_1d"]
        state["ventilation_index_1d"] = float(exact_vi)

        # 9. Rain event binary logic
        state["rain_event_1d"] = 1 if state["rainfall_1d"] >= 1.0 else 0

        # 10. Washout / extreme smog joint consistency
        if state["rainfall_1d"] > 30.0 and state["pm25"] > 250.0:
            notes.append(f"Washout suppression applied: PM2.5 reduced from {state['pm25']:.1f} to 90.0")
            state["pm25"] = 90.0
            was_corrected = True
            self._log_audit(trajectory_id, step_idx, "washout_smog_incoherence", "CORRECTED", state["pm25"])

        status = "CORRECTED" if was_corrected else "PASS"
        state["was_corrected"] = was_corrected
        state["correction_notes"] = "; ".join(notes) if notes else "PASS"

        return state, status, state["correction_notes"]

    def _log_audit(self, trajectory_id: str, step_idx: int, constraint_name: str, action: str, corrected_value: float):
        self.audit_records.append({
            "trajectory_id": trajectory_id,
            "step_idx": step_idx,
            "constraint_name": constraint_name,
            "action": action,
            "corrected_value": corrected_value,
        })

    def get_audit_dataframe(self) -> pd.DataFrame:
        if not self.audit_records:
            return pd.DataFrame(columns=["trajectory_id", "step_idx", "constraint_name", "action", "corrected_value"])
        return pd.DataFrame(self.audit_records)
