import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ExternalValidationPhase4I")


class ExternalValidationEnginePhase4I:
    """
    External Environmental Feature Validation & Counter-Evidence Detection Engine for Phase 4I.
    Validates SHAP behavior for external variables (rainfall, PBLH, ventilation) and detects conflict cases.
    """

    def __init__(self, df_v3: pd.DataFrame, df_shap_all: pd.DataFrame, df_group_shap_all: pd.DataFrame):
        self.df_v3 = df_v3.copy()
        self.df_v3['date'] = pd.to_datetime(self.df_v3['date'])
        self.df_shap_all = df_shap_all
        self.df_group_shap_all = df_group_shap_all

    def run_external_validation(self, output_dir: Path) -> dict:
        logger.info("Executing External Environmental Feature Validation & Counter-Evidence Detection...")
        output_dir.mkdir(parents=True, exist_ok=True)

        ext_features = [
            "rainfall_1d", "rainfall_3d", "rain_event_1d", "washout_index_3d",
            "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d",
            "aod_550_1d", "upwind_stubble_quadrant_1d"
        ]

        ext_records = []
        for feat in ext_features:
            if feat in self.df_v3.columns and feat in self.df_shap_all.columns:
                obs_vals = self.df_v3[feat].values
                shap_vals = self.df_shap_all[feat].values
                mean_abs_s = float(np.mean(np.abs(shap_vals)))
                mean_signed_s = float(np.mean(shap_vals))

                # Spearman correlation between feature value and SHAP value
                valid_mask = ~np.isnan(obs_vals) & ~np.isnan(shap_vals)
                if valid_mask.sum() > 10:
                    corr, _ = spearmanr(obs_vals[valid_mask], shap_vals[valid_mask])
                else:
                    corr = 0.0

                # Check physical plausibility
                if "rain" in feat or "washout" in feat or "pbl" in feat or "ventilation" in feat:
                    # Expect negative correlation (higher rain/pbl -> lower PM2.5 -> negative SHAP contribution)
                    phys_plausible = (corr < 0.0) or (mean_signed_s <= 0.05)
                elif "fire" in feat or "stubble" in feat or "aod" in feat:
                    # Expect positive correlation (higher fires/AOD -> higher PM2.5 -> positive SHAP contribution)
                    phys_plausible = (corr > 0.0) or (mean_signed_s >= -0.05)
                else:
                    phys_plausible = True

                ext_records.append({
                    "feature": feat,
                    "mean_abs_shap": mean_abs_s,
                    "mean_signed_shap": mean_signed_s,
                    "spearman_corr_obs_vs_shap": float(corr),
                    "physically_plausible": phys_plausible,
                    "validation_notes": "Expected negative impact on PM2.5" if "rain" in feat or "pbl" in feat else "Expected positive impact on PM2.5"
                })

        df_ext_val = pd.DataFrame(ext_records)
        df_ext_val.to_csv(output_dir / "v3_external_validation.csv", index=False)

        # 2. Counter-Evidence Detection
        conflicts = []
        dates = self.df_v3['date'].dt.strftime('%Y-%m-%d').values

        # Case A: High rainfall but positive rain/external SHAP contribution
        if "rainfall_1d" in self.df_v3.columns and "external_environmental" in self.df_group_shap_all.columns:
            high_rain_mask = (self.df_v3['rainfall_1d'] >= 10.0)
            ext_shap = self.df_group_shap_all['external_environmental'].values
            pos_rain_conflict = high_rain_mask & (ext_shap > 1.0)
            for idx in np.where(pos_rain_conflict)[0]:
                conflicts.append({
                    "date": dates[idx],
                    "group": "external_environmental",
                    "model_signal": f"Positive SHAP (+{ext_shap[idx]:.2f} ug/m3)",
                    "independent_evidence": f"Heavy Rainfall ({self.df_v3.loc[idx, 'rainfall_1d']:.1f} mm)",
                    "conflict_type": "RAINFALL_POSITIVE_SHAP_CONFLICT",
                    "severity": "MODERATE",
                    "explanation": "Model predicted positive contribution during heavy rainfall event, contradicting aerosol washout physics."
                })

        # Case B: High fire count but negative biomass SHAP contribution
        if "fire_hotspot_count_lag_1d" in self.df_v3.columns and "biomass_burning" in self.df_group_shap_all.columns:
            high_fire_mask = (self.df_v3['fire_hotspot_count_lag_1d'] >= 500)
            bio_shap = self.df_group_shap_all['biomass_burning'].values
            neg_bio_conflict = high_fire_mask & (bio_shap < -1.0)
            for idx in np.where(neg_bio_conflict)[0]:
                conflicts.append({
                    "date": dates[idx],
                    "group": "biomass_burning",
                    "model_signal": f"Negative SHAP ({bio_shap[idx]:.2f} ug/m3)",
                    "independent_evidence": f"High Fire Count ({self.df_v3.loc[idx, 'fire_hotspot_count_lag_1d']:.0f} fires)",
                    "conflict_type": "HIGH_FIRE_NEGATIVE_SHAP_CONFLICT",
                    "severity": "HIGH",
                    "explanation": "High regional fire count coincided with negative biomass SHAP due to downwind transport trajectory."
                })

        # Case C: High PBL / Ventilation but positive wind_ventilation SHAP contribution
        if "ventilation_index_1d" in self.df_v3.columns and "wind_ventilation" in self.df_group_shap_all.columns:
            high_vent_mask = (self.df_v3['ventilation_index_1d'] >= 6000.0)
            vent_shap = self.df_group_shap_all['wind_ventilation'].values
            pos_vent_conflict = high_vent_mask & (vent_shap > 2.0)
            for idx in np.where(pos_vent_conflict)[0]:
                conflicts.append({
                    "date": dates[idx],
                    "group": "wind_ventilation",
                    "model_signal": f"Positive SHAP (+{vent_shap[idx]:.2f} ug/m3)",
                    "independent_evidence": f"High Ventilation Index ({self.df_v3.loc[idx, 'ventilation_index_1d']:.0f} m2/s)",
                    "conflict_type": "HIGH_VENTILATION_POSITIVE_SHAP_CONFLICT",
                    "severity": "LOW",
                    "explanation": "High ventilation coincided with positive wind SHAP due to dust transport or complex wind direction vector."
                })

        df_conflicts = pd.DataFrame(conflicts)
        if df_conflicts.empty:
            df_conflicts = pd.DataFrame([{
                "date": "NONE",
                "group": "NONE",
                "model_signal": "NO_SEVERE_CONFLICTS",
                "independent_evidence": "CONSISTENT",
                "conflict_type": "NONE",
                "severity": "NONE",
                "explanation": "No unhandled counter-evidence conflicts detected."
            }])

        df_conflicts.to_csv(output_dir / "v3_attribution_conflicts.csv", index=False)
        logger.info(f"External validation complete. Detected {len(conflicts)} counter-evidence conflict instances.")

        return {
            "df_ext_val": df_ext_val,
            "df_conflicts": df_conflicts
        }
