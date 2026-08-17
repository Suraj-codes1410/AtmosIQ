"""
AtmosIQ Phase 8F: Physical Integrity & Hydrodynamic Invariant Auditor.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class Phase8FPhysicsAuditor:
    """Audits physical laws, boundary invariants, and hydrodynamic identities."""

    def audit_physics(self, df_8c: pd.DataFrame, df_8d: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        checks = []

        for name, df in [("AtmosIQ_Synthetic_Production_v1.0.0", df_8c), ("AtmosIQ_Synthetic_Calibrated_v0.1.0", df_8d)]:
            # 1. PM2.5 Non-Negativity
            neg_pm = int((df["pm25"] < 0.0).sum())
            checks.append({
                "corpus": name,
                "invariant": "PM2.5 Non-Negativity (>= 0 µg/m³)",
                "violations": neg_pm,
                "status": "PASS" if neg_pm == 0 else "FAIL",
                "details": f"Min PM2.5 observed: {df['pm25'].min():.2f} µg/m³",
            })

            # 2. Hydrodynamic Identity: VI = ws * PBLH
            ws_ms = df["wind_speed_kmh"] * (1000.0 / 3600.0)
            expected_vi = ws_ms * df["pblh_1d"]
            bad_vi = int((np.abs(df["ventilation_index_1d"] - expected_vi) > 1.0).sum())
            checks.append({
                "corpus": name,
                "invariant": "Hydrodynamic Identity (VI = ws_ms * PBLH)",
                "violations": bad_vi,
                "status": "PASS" if bad_vi == 0 else "FAIL",
                "details": f"Max VI residual: {np.max(np.abs(df['ventilation_index_1d'] - expected_vi)):.4e} m²/s",
            })

            # 3. Relative Humidity Bound [0, 100%]
            bad_rh = int(((df["humidity_pct"] < 0.0) | (df["humidity_pct"] > 100.0)).sum())
            checks.append({
                "corpus": name,
                "invariant": "Relative Humidity Bound [0, 100%]",
                "violations": bad_rh,
                "status": "PASS" if bad_rh == 0 else "FAIL",
                "details": f"Humidity range: [{df['humidity_pct'].min():.1f}%, {df['humidity_pct'].max():.1f}%]",
            })

            # 4. Rainfall Non-Negativity (>= 0 mm)
            bad_rain = int((df["rainfall_1d"] < 0.0).sum())
            checks.append({
                "corpus": name,
                "invariant": "Rainfall Non-Negativity (>= 0 mm)",
                "violations": bad_rain,
                "status": "PASS" if bad_rain == 0 else "FAIL",
                "details": f"Min Rainfall: {df['rainfall_1d'].min():.2f} mm",
            })

            # 5. Planetary Boundary Layer Height (>= 0 m)
            bad_pblh = int((df["pblh_1d"] < 0.0).sum())
            checks.append({
                "corpus": name,
                "invariant": "PBLH Non-Negativity (>= 0 m)",
                "violations": bad_pblh,
                "status": "PASS" if bad_pblh == 0 else "FAIL",
                "details": f"PBLH range: [{df['pblh_1d'].min():.1f}m, {df['pblh_1d'].max():.1f}m]",
            })

            # 6. Zero NaNs & ±Infs
            nan_count = int(df.isna().sum().sum())
            inf_count = int(np.isinf(df.select_dtypes(include=[np.number]).values).sum())
            checks.append({
                "corpus": name,
                "invariant": "Numerical Completeness (Zero NaN / ±Inf)",
                "violations": nan_count + inf_count,
                "status": "PASS" if (nan_count == 0 and inf_count == 0) else "FAIL",
                "details": f"Total NaNs: {nan_count}, Total Infs: {inf_count}",
            })

        df_aud = pd.DataFrame(checks)
        all_passed = bool((df_aud["status"] == "PASS").all())
        return all_passed, df_aud
