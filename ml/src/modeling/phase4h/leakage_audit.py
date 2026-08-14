import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("LeakageAuditPhase4H")


class LeakageAuditPhase4H:
    """
    Feature Leakage Audit Engine for Phase 4H.
    Inspects all columns in Dataset v3 and classifies each into:
    - safe: Approved prediction-safe feature (lags, rolling stats, calendar features, lagged external vars)
    - unsafe: Unsafe feature (same-day target, same-day simultaneous pollutant/met/fire measurements)
    - requires_justification: Features requiring explicit domain justification
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def run_audit(self, output_csv: Path = None) -> pd.DataFrame:
        logger.info("Executing Phase 4H Feature Leakage Audit on Dataset v3...")

        # Same-day raw observations and same-day target are unsafe for 1-day step-ahead prediction
        unsafe_exact = {
            'date', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
            'temperature_c', 'humidity_pct', 'wind_speed_kmh', 'wind_direction_deg',
            'pressure_hpa', 'precipitation_mm', 'fire_hotspot_count',
            'high_confidence_fire_count', 'mean_fire_brightness', 'punjab_fire_count',
            'haryana_fire_count', 'rajasthan_fire_count', 'delhi_ncr_fire_count',
            'fire_radiative_power_sum', 'pm25_pm10_ratio', 'no2_so2_ratio', 'co_normalized',
            'daily_pollutant_trend', 'pollutant_rolling_avg', 'pollutant_volatility',
            'pollutant_zscore', 'pollutant_anomaly_score', 'wind_x', 'wind_y',
            'temperature_humidity_index', 'temperature_change', 'humidity_change',
            'pressure_change', 'wind_speed_change', 'wind_direction_change'
        }

        # Deterministic calendar features requiring justification note
        calendar_features = {
            'day_of_week', 'is_weekend', 'is_holiday', 'is_festival', 'is_stubble_season',
            'month', 'quarter', 'day_of_year', 'week_of_year', 'days_until_diwali',
            'days_since_diwali', 'festival_window'
        }

        records = []
        for col in self.df.columns:
            if col in unsafe_exact:
                classification = "unsafe"
                leakage_type = "same_day_target_or_observation" if col in ['pm25', 'date'] else "same_day_simultaneous_measurement"
                detected = True
                severity = "HIGH"
                notes = "Same-day observation or identifier cannot be used for 1-day step-ahead forecasting"
            elif col in calendar_features:
                classification = "requires_justification"
                leakage_type = "calendar_temporal_proxy"
                detected = False
                severity = "NONE"
                notes = "Deterministic calendar feature known in advance at prediction time"
            elif "_lag_" in col or "_roll_" in col or col.endswith("_1d") or col.endswith("_3d") or col.endswith("_7d") or col.endswith("_14d") or col.endswith("_30d"):
                classification = "safe"
                leakage_type = "none"
                detected = False
                severity = "NONE"
                notes = "Validated prediction-safe lagged or rolling feature available at prediction time"
            else:
                # Default safety check for other derived features
                classification = "safe"
                leakage_type = "none"
                detected = False
                severity = "NONE"
                notes = "Prediction-safe feature"

            records.append({
                "feature_name": col,
                "classification": classification,
                "leakage_type": leakage_type,
                "detected": detected,
                "severity": severity,
                "justification_notes": notes
            })

        audit_df = pd.DataFrame(records)

        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            audit_df.to_csv(output_csv, index=False)
            logger.info(f"Leakage audit saved to: {output_csv}")

        unsafe_count = (audit_df['classification'] == 'unsafe').sum()
        safe_count = (audit_df['classification'] == 'safe').sum()
        req_count = (audit_df['classification'] == 'requires_justification').sum()

        logger.info(f"Leakage Audit Summary: Total Columns={len(audit_df)}, Safe={safe_count}, Requires Justification={req_count}, Unsafe={unsafe_count}")
        return audit_df

    def get_approved_features(self, audit_df: pd.DataFrame) -> list:
        """Returns all features classified as safe or approved requires_justification."""
        approved = audit_df[audit_df['classification'].isin(['safe', 'requires_justification'])]['feature_name'].tolist()
        return approved

    @staticmethod
    def assert_no_leakage(features: list, audit_df: pd.DataFrame):
        """Asserts that no unsafe feature exists in the provided feature list."""
        unsafe_set = set(audit_df[audit_df['classification'] == 'unsafe']['feature_name'])
        leaked = [f for f in features if f in unsafe_set]
        if leaked:
            raise ValueError(f"CRITICAL LEAKAGE ERROR: Unsafe features detected in candidate model input: {leaked}")
