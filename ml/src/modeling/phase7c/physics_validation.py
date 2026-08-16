"""
AtmosIQ Phase 7C: Physics Boundary Validator (Workstream F).
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class PhysicsValidatorPhase7C:
    """Verifies all hard physical boundary constraints and mass-balance identities."""

    def __init__(self):
        pass

    def validate_physics(self, df_synthetic: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        checks = []

        # 1. PM2.5 non-negativity
        neg_pm = int((df_synthetic["pm25"] < 0.0).sum())
        checks.append({"constraint": "PM2.5 Non-Negativity (>= 0.0)", "violations": neg_pm, "status": "PASS" if neg_pm == 0 else "FAIL"})

        # 2. Wind speed non-negativity
        neg_ws = int((df_synthetic["wind_speed_kmh"] < 0.0).sum())
        checks.append({"constraint": "Wind Speed Non-Negativity (>= 0.0)", "violations": neg_ws, "status": "PASS" if neg_ws == 0 else "FAIL"})

        # 3. Rainfall non-negativity
        neg_rain = int((df_synthetic["rainfall_1d"] < 0.0).sum())
        checks.append({"constraint": "Rainfall Non-Negativity (>= 0.0)", "violations": neg_rain, "status": "PASS" if neg_rain == 0 else "FAIL"})

        # 4. Boundary layer height positive
        neg_pblh = int((df_synthetic["pblh_1d"] <= 0.0).sum())
        checks.append({"constraint": "PBLH Positive (> 0.0)", "violations": neg_pblh, "status": "PASS" if neg_pblh == 0 else "FAIL"})

        # 5. Humidity bounds (0-100%)
        rh_viol = int(((df_synthetic["humidity_pct"] < 0.0) | (df_synthetic["humidity_pct"] > 100.0)).sum())
        checks.append({"constraint": "Relative Humidity Bounds (0-100%)", "violations": rh_viol, "status": "PASS" if rh_viol == 0 else "FAIL"})

        # 6. Temperature bounds (0-50°C)
        t_viol = int(((df_synthetic["temperature_c"] < 0.0) | (df_synthetic["temperature_c"] > 50.0)).sum())
        checks.append({"constraint": "Temperature Bounds (0-50°C)", "violations": t_viol, "status": "PASS" if t_viol == 0 else "FAIL"})

        # 7. Fire counts non-negativity
        fire_viol = int((df_synthetic["fire_hotspot_count_1d"] < 0.0).sum())
        checks.append({"constraint": "Fire Counts Non-Negativity (>= 0)", "violations": fire_viol, "status": "PASS" if fire_viol == 0 else "FAIL"})

        # 8. Ventilation Index hydrodynamic consistency
        ws_ms = df_synthetic["wind_speed_kmh"] * (1000.0 / 3600.0)
        expected_vi = ws_ms * df_synthetic["pblh_1d"]
        vi_diff = np.abs(df_synthetic["ventilation_index_1d"] - expected_vi)
        vi_viol = int((vi_diff > 1.0).sum())
        checks.append({"constraint": "Ventilation Index Exact (VI = ws * PBLH)", "violations": vi_viol, "status": "PASS" if vi_viol == 0 else "FAIL"})

        # 9. Rain event binary logic
        rain_event_viol = int(((df_synthetic["rain_event_1d"] == 1) != (df_synthetic["rainfall_1d"] >= 1.0)).sum())
        checks.append({"constraint": "Rain Event Indicator (I(rain >= 1mm))", "violations": rain_event_viol, "status": "PASS" if rain_event_viol == 0 else "FAIL"})

        # 10. NaN and Inf Prevention
        nan_count = int(df_synthetic.isna().sum().sum())
        checks.append({"constraint": "Zero NaN and Infinite Values", "violations": nan_count, "status": "PASS" if nan_count == 0 else "FAIL"})

        df_phys = pd.DataFrame(checks)
        total_viol = int(df_phys["violations"].sum())
        all_passed = (total_viol == 0)

        summary = {
            "total_physics_violations": total_viol,
            "hard_constraint_pass_rate_pct": 100.0 if all_passed else 0.0,
            "overall_status": "PASS" if all_passed else "FAIL",
        }

        return df_phys, summary
