import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("CounterfactualEnginePhase4I")


class CounterfactualRevalidationEnginePhase4I:
    """
    Counterfactual Simulation, Plausibility, and SHAP-Consistency Engine for Phase 4I.
    Recomputes counterfactual predictions against the promoted v3 Random Forest model.
    """

    SCENARIOS = [
        "biomass_low", "biomass_median", "biomass_high",
        "wind_stagnant", "wind_normal", "wind_dispersion",
        "meteorology_normal",
        "combined_biomass_wind", "combined_all_favorable"
    ]

    def __init__(self, model, df_v3: pd.DataFrame, features_35: list, feature_to_group: dict, df_group_shap_all: pd.DataFrame):
        self.model = model
        self.df_v3 = df_v3.copy()
        self.df_v3['date'] = pd.to_datetime(self.df_v3['date'])
        self.features = features_35
        self.feature_to_group = feature_to_group
        self.df_group_shap_all = df_group_shap_all

        self.X_base = self.df_v3[self.features].fillna(0.0)
        self.y_obs_pred = self.model.predict(self.X_base)

        # Precompute reference quantiles (10th, 50th, 90th)
        self.quantiles = {
            col: {
                "q10": float(self.X_base[col].quantile(0.10)),
                "q50": float(self.X_base[col].quantile(0.50)),
                "q90": float(self.X_base[col].quantile(0.90))
            }
            for col in self.features
        }

    def _apply_scenario_intervention(self, X_orig: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, list]:
        X_cf = X_orig.copy()
        modified_cols = []

        if scenario == "biomass_low":
            for c in ["fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d", "fire_hotspot_count_roll_mean_7d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q10"]
                    modified_cols.append(c)
            if "is_stubble_season" in X_cf.columns:
                X_cf["is_stubble_season"] = 0
                modified_cols.append("is_stubble_season")

        elif scenario == "biomass_median":
            for c in ["fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d", "fire_hotspot_count_roll_mean_7d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q50"]
                    modified_cols.append(c)

        elif scenario == "biomass_high":
            for c in ["fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d", "fire_hotspot_count_roll_mean_7d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q90"]
                    modified_cols.append(c)
            if "is_stubble_season" in X_cf.columns:
                X_cf["is_stubble_season"] = 1
                modified_cols.append("is_stubble_season")

        elif scenario == "wind_stagnant":
            for c in ["wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d", "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q10"]
                    modified_cols.append(c)

        elif scenario == "wind_normal":
            for c in ["wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d", "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q50"]
                    modified_cols.append(c)

        elif scenario == "wind_dispersion":
            for c in ["wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d", "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q90"]
                    modified_cols.append(c)

        elif scenario == "meteorology_normal":
            for c in ["temperature_c_lag_1d", "temperature_c_roll_mean_3d", "humidity_pct_lag_1d", "humidity_pct_roll_mean_3d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q50"]
                    modified_cols.append(c)

        elif scenario == "combined_biomass_wind":
            X_cf, mod1 = self._apply_scenario_intervention(X_cf, "biomass_low")
            X_cf, mod2 = self._apply_scenario_intervention(X_cf, "wind_dispersion")
            modified_cols = list(set(mod1 + mod2))

        elif scenario == "combined_all_favorable":
            X_cf, mod1 = self._apply_scenario_intervention(X_cf, "combined_biomass_wind")
            for c in ["rainfall_1d", "rainfall_3d", "washout_index_3d"]:
                if c in X_cf.columns:
                    X_cf[c] = self.quantiles[c]["q90"]
                    modified_cols.append(c)
            if "rain_event_1d" in X_cf.columns:
                X_cf["rain_event_1d"] = 1
                modified_cols.append("rain_event_1d")
            modified_cols = list(set(mod1 + modified_cols))

        return X_cf, modified_cols

    def run_counterfactual_revalidation(self, output_dir: Path) -> dict:
        logger.info("Executing Counterfactual Revalidation, Plausibility, and SHAP-Consistency Engines...")
        output_dir.mkdir(parents=True, exist_ok=True)

        cf_results = []
        plausibility_records = []
        dates = self.df_v3['date'].dt.strftime('%Y-%m-%d').values
        actual_pm25 = self.df_v3['pm25'].values

        # Mean & Std for OOD standardization
        X_mean = self.X_base.mean()
        X_std = self.X_base.std().replace(0.0, 1.0)

        for sc in self.SCENARIOS:
            X_cf, mod_cols = self._apply_scenario_intervention(self.X_base, sc)
            
            # Predict
            y_cf_pred = self.model.predict(X_cf)
            delta_y = y_cf_pred - self.y_obs_pred

            # Plausibility check: feature isolation
            unmod_cols = [c for c in self.features if c not in mod_cols]
            isolation_pass = (X_cf[unmod_cols].values == self.X_base[unmod_cols].values).all()

            # Non-NaN & finite check
            finite_pass = np.isfinite(y_cf_pred).all() and not np.isnan(y_cf_pred).any()

            # OOD distance (standardized Euclidean distance per sample)
            z_scores = (X_cf - X_mean) / X_std
            ood_distances = np.sqrt((z_scores ** 2).mean(axis=1))
            max_ood = float(np.max(ood_distances))
            mean_ood = float(np.mean(ood_distances))

            plausible_pass = isolation_pass and finite_pass and (max_ood <= 5.0)

            plausibility_records.append({
                "scenario": sc,
                "num_intervened_features": len(mod_cols),
                "feature_isolation_pass": isolation_pass,
                "prediction_validity_pass": finite_pass,
                "mean_ood_distance": mean_ood,
                "max_ood_distance": max_ood,
                "overall_plausibility_pass": plausible_pass,
                "status": "PASS" if plausible_pass else "FAIL"
            })

            # Record per-observation scenario results summary
            mean_delta = float(np.mean(delta_y))
            mean_cf_pred = float(np.mean(y_cf_pred))
            mean_obs_pred = float(np.mean(self.y_obs_pred))

            cf_results.append({
                "scenario": sc,
                "mean_observed_prediction": mean_obs_pred,
                "mean_counterfactual_prediction": mean_cf_pred,
                "mean_delta_pm25": mean_delta,
                "median_delta_pm25": float(np.median(delta_y)),
                "min_delta_pm25": float(np.min(delta_y)),
                "max_delta_pm25": float(np.max(delta_y)),
                "intervened_features_count": len(mod_cols)
            })

        df_cf_summary = pd.DataFrame(cf_results)
        df_cf_summary.to_csv(output_dir / "v3_counterfactual_results.csv", index=False)

        df_plausible = pd.DataFrame(plausibility_records)
        df_plausible.to_csv(output_dir / "v3_counterfactual_plausibility.csv", index=False)

        # 3. SHAP vs Counterfactual Consistency Evaluation (Active-day physical agreement)
        consistency_records = []

        # Group: biomass_burning (biomass_low reduces PM2.5 -> expecting negative delta_y on active biomass days SHAP > 1.0)
        X_bio_low, _ = self._apply_scenario_intervention(self.X_base, "biomass_low")
        y_bio_pred = self.model.predict(X_bio_low)
        delta_bio = y_bio_pred - self.y_obs_pred
        shap_bio = self.df_group_shap_all['biomass_burning'].values if 'biomass_burning' in self.df_group_shap_all.columns else np.zeros(len(self.df_v3))

        active_bio_mask = (shap_bio > 1.0)
        bio_agreements = (delta_bio[active_bio_mask] < 0.0) if active_bio_mask.sum() > 0 else np.array([True])
        bio_rate = float(bio_agreements.mean())

        consistency_records.append({
            "group": "biomass_burning",
            "benchmark_scenario": "biomass_low",
            "active_obs_count": int(active_bio_mask.sum()),
            "directional_consistency_rate": bio_rate,
            "v2_historical_benchmark": 0.944,
            "status": "PASS" if bio_rate >= 0.85 else "WARN"
        })

        # Group: wind_ventilation (wind_stagnant increases PM2.5 -> expecting positive delta_y on active wind days SHAP < -1.0)
        X_wind_stag, _ = self._apply_scenario_intervention(self.X_base, "wind_stagnant")
        y_wind_pred = self.model.predict(X_wind_stag)
        delta_wind = y_wind_pred - self.y_obs_pred
        shap_wind = self.df_group_shap_all['wind_ventilation'].values if 'wind_ventilation' in self.df_group_shap_all.columns else np.zeros(len(self.df_v3))

        active_wind_mask = (shap_wind < -1.0)
        wind_agreements = (delta_wind[active_wind_mask] > 0.0) if active_wind_mask.sum() > 0 else np.array([True])
        wind_rate = float(wind_agreements.mean())

        consistency_records.append({
            "group": "wind_ventilation",
            "benchmark_scenario": "wind_stagnant",
            "active_obs_count": int(active_wind_mask.sum()),
            "directional_consistency_rate": wind_rate,
            "v2_historical_benchmark": 0.944,
            "status": "PASS" if wind_rate >= 0.85 else "WARN"
        })

        df_consistency = pd.DataFrame(consistency_records)
        df_consistency.to_csv(output_dir / "v3_shap_counterfactual_consistency.csv", index=False)

        logger.info("Counterfactual revalidation & consistency check completed.")
        return {
            "df_cf_summary": df_cf_summary,
            "df_plausible": df_plausible,
            "df_consistency": df_consistency
        }
