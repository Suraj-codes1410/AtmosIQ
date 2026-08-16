"""
AtmosIQ Phase 7B: Simplified Bulk Atmospheric Mass-Balance and Dispersion Model.
"""

import numpy as np
from typing import Dict, Any, Tuple


class AtmosphericMassBalanceModel:
    """
    Simplified physics-informed bulk PM2.5 mass-balance evolution:
    dC/dt = (E_anthro + E_fire) / PBLH - k_disp(ws, PBLH) * C - k_washout(Rain) * C + eps
    """

    def __init__(self, random_state: np.random.RandomState):
        self.rng = random_state

    def step_mass_balance(
        self,
        prev_pm25: float,
        season: str,
        regime: str,
        met_sample: Dict[str, float],
        is_stubble_season: int,
        festival_window: int,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates next PM2.5, wind vectors, ventilation index, fire hotspot count, and upwind transport score.
        """
        temp_c = met_sample["temperature_c"]
        humidity_pct = met_sample["humidity_pct"]
        ws_kmh = max(met_sample["wind_speed_kmh"], 1.0)
        pblh = max(met_sample["pblh_1d"], 200.0)
        pblh_min = max(met_sample.get("pblh_min_1d", pblh * 0.65), 150.0)
        rainfall = max(met_sample["rainfall_1d"], 0.0)

        # 1. Ventilation Index (m²/s) = (ws in m/s) * PBLH (m)
        ws_ms = ws_kmh * (1000.0 / 3600.0)
        ventilation_index = ws_ms * pblh

        # 2. Wind Direction & Vector Decomposition
        # Post-Monsoon / Winter typically has NW winds (315°), Summer has W/SW (240°)
        if season in ["Post-Monsoon", "Winter"]:
            base_deg = 315.0 + self.rng.normal(0, 20.0)
        elif season == "Monsoon":
            base_deg = 120.0 + self.rng.normal(0, 30.0)
        else:
            base_deg = 270.0 + self.rng.normal(0, 25.0)

        rad = np.radians(base_deg)
        # u = -ws * sin(deg), v = -ws * cos(deg)
        wind_u = -ws_ms * np.sin(rad)
        wind_v = -ws_ms * np.cos(rad)

        # 3. Satellite Active Fire Hotspot Generation
        if is_stubble_season:
            # Elevated fire counts conditioned on regime
            if regime == "Extreme":
                fire_count = float(np.clip(self.rng.gamma(shape=4.0, scale=80.0), 50.0, 1200.0))
            elif regime == "High":
                fire_count = float(np.clip(self.rng.gamma(shape=2.5, scale=40.0), 10.0, 500.0))
            else:
                fire_count = float(np.clip(self.rng.gamma(shape=1.5, scale=20.0), 2.0, 150.0))
            
            # Transport alignment score (NW quadrant: wind_u > 0, wind_v < 0)
            nw_alignment = max(0.0, (wind_u - wind_v) / (np.sqrt(wind_u**2 + wind_v**2) + 1e-4))
            upwind_score = float(np.clip(nw_alignment * (fire_count / 30.0), 0.0, 35.0))
        else:
            fire_count = float(np.clip(self.rng.exponential(scale=15.0), 1.0, 80.0))
            upwind_score = float(np.clip(fire_count / 100.0, 0.0, 5.0))

        # 4. Anthropogenic & Fire Emission Forcing
        # Baseline emissions modulated by season and festival window
        e_base = 12000.0  # µg / (m² · day)
        if season == "Winter":
            e_base *= 1.35
        elif season == "Monsoon":
            e_base *= 0.70
        if festival_window:
            e_base *= 2.20

        e_fire = (fire_count * 150.0) if is_stubble_season else 0.0
        total_emission = e_base + e_fire

        # 5. Dispersion & Washout Rate Constants
        # Stagnant when VI is low; high dispersion when VI is high
        k_disp = float(np.clip(0.00015 * ventilation_index + 0.10, 0.15, 1.80))
        k_washout = float(np.clip(0.08 * rainfall, 0.0, 2.50))

        # 6. Bulk Mass-Balance Step
        # delta_C = E / PBLH - (k_disp + k_washout) * C_prev + noise
        c_source = total_emission / pblh
        dc = c_source - (k_disp + k_washout) * (prev_pm25 / 2.0)
        
        # Add stochastic regime innovation
        innovation = met_sample.get("pm25_delta", 0.0) * 0.40
        next_pm25 = float(prev_pm25 + dc * 0.45 + innovation)

        # 7. Aerosol Optical Depth (AOD) Coupling
        aod = float(np.clip(0.0028 * next_pm25 + 0.12 + self.rng.normal(0, 0.05), 0.08, 1.45))

        return next_pm25, wind_u, wind_v, ventilation_index, fire_count, upwind_score, aod
