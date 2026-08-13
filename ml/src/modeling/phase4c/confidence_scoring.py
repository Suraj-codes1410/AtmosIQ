import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("ConfidenceScoringPhase4C")


class ConfidenceScoringPhase4C:
    """
    AtmosIQ Phase 4C Attribution Confidence & Conflict Evaluator.
    Calculates 0-3 evidence support scores, assigns confidence levels (Low, Moderate, High), and identifies counter-evidence conflict cases.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_confidence_and_conflicts(self, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculates confidence scores and flags attribution conflict cases."""
        logger.info("Executing Attribution Confidence Scoring & Counter-Evidence Conflict Detection...")

        fire_col = "fire_hotspot_count_lag_1d" if "fire_hotspot_count_lag_1d" in df.columns else "fire_hotspot_count_roll_mean_7d"
        wind_col = "wind_speed_kmh_lag_1d" if "wind_speed_kmh_lag_1d" in df.columns else "wind_speed_kmh_roll_mean_7d"
        temp_col = "temperature_c_roll_mean_3d" if "temperature_c_roll_mean_3d" in df.columns else "temperature_c_lag_1d"

        fire_q75 = float(df[fire_col].quantile(0.75))
        fire_q25 = float(df[fire_col].quantile(0.25))
        wind_low = 5.0
        wind_high = 12.0
        temp_q25 = float(df[temp_col].quantile(0.25))

        conf_rows = []
        conflict_rows = []

        for i in range(len(df)):
            dt = df.loc[i, "date"]
            act_pm25 = float(df.loc[i, "pm25"])
            pred_pm25 = float(group_shap_df.loc[i, "predicted_pm25"])

            bio_shap = float(group_shap_df.loc[i, "biomass_burning_shap"])
            wind_shap = float(group_shap_df.loc[i, "wind_ventilation_shap"])
            met_shap = float(group_shap_df.loc[i, "meteorology_shap"])
            pers_shap = float(group_shap_df.loc[i, "pm25_persistence_shap"])

            fire_v = float(df.loc[i, fire_col])
            wind_v = float(df.loc[i, wind_col])
            temp_v = float(df.loc[i, temp_col])

            score = 0
            # 1. Biomass evidence
            if fire_v >= fire_q75 and bio_shap > 0:
                score += 1

            # 2. Ventilation evidence
            if wind_v <= wind_low and wind_shap > 0:
                score += 1

            # 3. Meteorological evidence
            if temp_v <= temp_q25 and met_shap > 0:
                score += 1

            if score == 0:
                conf_level = "Low"
            elif score == 1:
                conf_level = "Moderate"
            else:
                conf_level = "High"

            conf_rows.append({
                "date": dt,
                "actual_pm25": act_pm25,
                "predicted_pm25": pred_pm25,
                "evidence_score": score,
                "confidence_level": conf_level,
                "biomass_burning_shap": bio_shap,
                "wind_ventilation_shap": wind_shap,
                "meteorology_shap": met_shap,
                "pm25_persistence_shap": pers_shap
            })

            # Counter-evidence conflict checks
            # Conflict A: High biomass SHAP (>75th percentile of biomass SHAP) but low fire activity (<=25th percentile)
            bio_shap_q75 = float(group_shap_df["biomass_burning_shap"].quantile(0.75))
            if bio_shap >= bio_shap_q75 and fire_v <= fire_q25:
                conflict_rows.append({
                    "date": dt,
                    "attribution_group": "biomass_burning",
                    "shap_value": bio_shap,
                    "environmental_indicator": fire_col,
                    "indicator_value": fire_v,
                    "expected_relationship": "High biomass SHAP requires high upwind fire count",
                    "observed_relationship": "High biomass SHAP co-occurs with low upwind fire count",
                    "conflict_type": "high_shap_low_fire_activity"
                })

            # Conflict B: High fire activity (>=75th percentile) but negative biomass SHAP
            if fire_v >= fire_q75 and bio_shap < -1.0:
                conflict_rows.append({
                    "date": dt,
                    "attribution_group": "biomass_burning",
                    "shap_value": bio_shap,
                    "environmental_indicator": fire_col,
                    "indicator_value": fire_v,
                    "expected_relationship": "High upwind fire count requires positive biomass SHAP",
                    "observed_relationship": "High fire count co-occurs with negative biomass SHAP",
                    "conflict_type": "high_fire_negative_biomass_shap"
                })

            # Conflict C: Positive wind SHAP (>5 µg/m³) during high surface wind speed (>=12 km/h)
            if wind_v >= wind_high and wind_shap > 5.0:
                conflict_rows.append({
                    "date": dt,
                    "attribution_group": "wind_ventilation",
                    "shap_value": wind_shap,
                    "environmental_indicator": wind_col,
                    "indicator_value": wind_v,
                    "expected_relationship": "Positive ventilation SHAP requires low surface wind speed",
                    "observed_relationship": "Positive ventilation SHAP co-occurs with high wind speed",
                    "conflict_type": "positive_ventilation_shap_high_wind"
                })

        conf_df = pd.DataFrame(conf_rows)
        conflict_df = pd.DataFrame(conflict_rows)

        conf_df.to_csv(self.exp_dir / "confidence_scores.csv", index=False)
        conflict_df.to_csv(self.exp_dir / "attribution_conflicts.csv", index=False)

        high_conf_pct = float(np.mean(conf_df["confidence_level"] == "High")) * 100
        mod_conf_pct = float(np.mean(conf_df["confidence_level"] == "Moderate")) * 100
        logger.info(f"Confidence scoring complete -> High: {high_conf_pct:.1f}%, Moderate: {mod_conf_pct:.1f}%. Identified {len(conflict_df)} counter-evidence conflict cases.")

        return conf_df, conflict_df


if __name__ == "__main__":
    scorer = ConfidenceScoringPhase4C()
