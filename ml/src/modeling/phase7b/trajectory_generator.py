"""
AtmosIQ Phase 7B: Main Stochastic Trajectory Generator (HP-STG).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import datetime

from .config import SyntheticConfigPhase7B
from .state import AtmosphericState, TrajectoryBatch
from .regime_model import RegimeMarkovModel
from .seasonal_model import SeasonalCalendarModel
from .physics_model import AtmosphericMassBalanceModel
from .stochastic_process import CorrelatedInnovationSampler
from .constraint_engine import PhysicsConstraintEnginePhase7B
from .feature_reconstruction import FeatureReconstructorPhase7B


class TrajectoryGeneratorPhase7B:
    """
    Orchestrates the generation of sequential physics-informed atmospheric trajectories.
    """

    def __init__(self, config: SyntheticConfigPhase7B, feature_registry: List[str]):
        self.config = config
        self.feature_registry = list(feature_registry)
        self.rng = np.random.RandomState(config.random_seed)

        self.regime_model = RegimeMarkovModel(self.rng)
        self.seasonal_model = SeasonalCalendarModel(self.rng)
        self.physics_model = AtmosphericMassBalanceModel(self.rng)
        self.innovation_sampler = CorrelatedInnovationSampler(self.rng)
        self.constraint_engine = PhysicsConstraintEnginePhase7B()
        self.reconstructor = FeatureReconstructorPhase7B(self.feature_registry)

    def fit_from_training_data(self, df_train: pd.DataFrame):
        """Fit empirical models strictly from the 2020-2021 training partition."""
        self.regime_model.fit_from_training_data(df_train)
        self.seasonal_model.fit_from_training_data(df_train)
        self.innovation_sampler.fit_from_training_data(df_train)

    def generate_single_trajectory(
        self,
        trajectory_id: str,
        length: int,
        season: str,
        target_regime: str = None
    ) -> pd.DataFrame:
        """Generates a single continuous synthetic trajectory of specified length."""
        regime_seq = self.regime_model.sample_regime_sequence(length, season, initial_regime=target_regime)
        states: List[AtmosphericState] = []

        # Initial baseline PM2.5 based on starting regime
        if regime_seq[0] == "Low": prev_pm25 = float(self.rng.uniform(25.0, 50.0))
        elif regime_seq[0] == "Moderate": prev_pm25 = float(self.rng.uniform(70.0, 110.0))
        elif regime_seq[0] == "High": prev_pm25 = float(self.rng.uniform(140.0, 220.0))
        else: prev_pm25 = float(self.rng.uniform(260.0, 360.0))

        gen_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for step in range(length):
            regime = regime_seq[step]
            cal_ctx = self.seasonal_model.get_seasonal_context(season, step)
            is_stubble = cal_ctx["is_stubble_season"]
            festival = cal_ctx["festival_window"]
            priors = cal_ctx["priors"]

            # Sample base weather conditioned on season & regime innovation
            innov = self.innovation_sampler.sample_innovation(regime)
            t_mean, t_std = priors.get("temperature_c", (25.0, 5.0))
            rh_mean, rh_std = priors.get("humidity_pct", (60.0, 15.0))
            ws_mean, ws_std = priors.get("wind_speed_kmh", (15.0, 5.0))
            pblh_mean, pblh_std = priors.get("pblh_1d", (1200.0, 350.0))

            temp_c = float(np.clip(t_mean + innov["temperature_c"] * 0.35 + self.rng.normal(0, 1.5), 5.0, 45.0))
            humidity = float(np.clip(rh_mean + innov["humidity_pct"] * 0.35 + self.rng.normal(0, 3.0), 10.0, 95.0))
            
            # If extreme regime in winter, simulate stagnant surface conditions
            if regime == "Extreme" and season in ["Winter", "Post-Monsoon"]:
                ws = float(np.clip(self.rng.uniform(3.5, 9.5), 2.0, 12.0))
                pblh = float(np.clip(self.rng.uniform(350.0, 680.0), 200.0, 800.0))
                pblh_min = float(np.clip(pblh * self.rng.uniform(0.40, 0.65), 150.0, 450.0))
                rain = 0.0
            else:
                ws = float(max(ws_mean + innov["wind_speed_kmh"] * 0.40, 3.0))
                pblh = float(max(pblh_mean + innov["pblh_1d"] * 0.40, 350.0))
                pblh_min = float(max(pblh * 0.65, 180.0))
                rain = float(max(innov["rainfall_1d"], 0.0) if season == "Monsoon" else 0.0)

            met_sample = {
                "temperature_c": temp_c,
                "humidity_pct": humidity,
                "wind_speed_kmh": ws,
                "pblh_1d": pblh,
                "pblh_min_1d": pblh_min,
                "rainfall_1d": rain,
                "pm25_delta": innov.get("pm25_delta", 0.0),
            }

            # Physics mass-balance step
            next_pm, wind_u, wind_v, vi, fire_cnt, upwind_sc, aod = self.physics_model.step_mass_balance(
                prev_pm25, season, regime, met_sample, is_stubble, festival
            )

            raw_state = {
                "step_idx": step,
                "synthetic_date": f"SYNTH-{trajectory_id}-D{step+1:03d}",
                "season": season,
                "pollution_regime": regime,
                "pm25": next_pm,
                "temperature_c": temp_c,
                "humidity_pct": humidity,
                "wind_speed_kmh": ws,
                "wind_u_component_1d": wind_u,
                "wind_v_component_1d": wind_v,
                "pblh_1d": pblh,
                "pblh_min_1d": pblh_min,
                "rainfall_1d": rain,
                "rain_event_1d": 1 if rain >= 1.0 else 0,
                "ventilation_index_1d": vi,
                "aod_550_1d": aod,
                "fire_hotspot_count_1d": fire_cnt,
                "upwind_stubble_quadrant_1d": upwind_sc,
                "is_stubble_season": is_stubble,
                "festival_window": festival,
            }

            # Enforce physical constraints
            constrained_state, status, notes = self.constraint_engine.evaluate_and_constrain(
                raw_state, trajectory_id, step
            )

            state_obj = AtmosphericState(
                step_idx=step,
                synthetic_date=constrained_state["synthetic_date"],
                season=season,
                pollution_regime=regime,
                pm25=constrained_state["pm25"],
                temperature_c=constrained_state["temperature_c"],
                humidity_pct=constrained_state["humidity_pct"],
                wind_speed_kmh=constrained_state["wind_speed_kmh"],
                wind_u_component_1d=constrained_state["wind_u_component_1d"],
                wind_v_component_1d=constrained_state["wind_v_component_1d"],
                pblh_1d=constrained_state["pblh_1d"],
                pblh_min_1d=constrained_state["pblh_min_1d"],
                rainfall_1d=constrained_state["rainfall_1d"],
                rain_event_1d=constrained_state["rain_event_1d"],
                ventilation_index_1d=constrained_state["ventilation_index_1d"],
                aod_550_1d=constrained_state["aod_550_1d"],
                fire_hotspot_count_1d=constrained_state["fire_hotspot_count_1d"],
                upwind_stubble_quadrant_1d=constrained_state["upwind_stubble_quadrant_1d"],
                is_stubble_season=is_stubble,
                festival_window=festival,
                was_corrected=constrained_state["was_corrected"],
                correction_notes=notes,
            )

            states.append(state_obj)
            prev_pm25 = constrained_state["pm25"]

        batch = TrajectoryBatch(
            trajectory_id=trajectory_id,
            length=length,
            season=season,
            target_regime=target_regime or regime_seq[0],
            states=states,
        )

        df_raw = batch.to_dataframe()
        # Reconstruct all 35 features
        df_reconstructed = self.reconstructor.reconstruct_trajectory_features(df_raw)

        # Attach explicit synthetic provenance fields
        df_reconstructed["data_origin"] = self.config.data_origin
        df_reconstructed["generator_version"] = f"{self.config.generator_name} v{self.config.generator_version}"
        df_reconstructed["synthetic_timestamp"] = df_reconstructed["synthetic_date"]
        df_reconstructed["source_spec_hash"] = self.config.phase7a_spec_path.stem
        df_reconstructed["random_seed"] = self.config.random_seed
        df_reconstructed["generation_timestamp"] = gen_ts

        return df_reconstructed

    def generate_all_trajectories(self) -> pd.DataFrame:
        """Generates all configured trajectories across seasons and lengths."""
        all_dfs = []
        seasons = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
        traj_count = 0

        for length, num_trajs in self.config.num_trajectories_per_length.items():
            for i in range(num_trajs):
                traj_count += 1
                season = seasons[i % len(seasons)]
                # Ensure post-monsoon and winter get extreme regimes
                target_regime = "Extreme" if (season in ["Winter", "Post-Monsoon"] and i % 2 == 0) else None
                traj_id = f"T{traj_count:03d}_L{length}_{season[:3].upper()}"
                
                df_traj = self.generate_single_trajectory(traj_id, length, season, target_regime)
                all_dfs.append(df_traj)

        df_all = pd.concat(all_dfs, ignore_index=True)
        return df_all
