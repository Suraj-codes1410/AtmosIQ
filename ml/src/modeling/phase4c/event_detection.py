import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("EventDetectionPhase4C")


class EventDetectionPhase4C:
    """
    AtmosIQ Phase 4C Extreme Pollution Event Detector.
    Identifies high-pollution episodes (PM2.5 >= 90th percentile), groups contiguous exceedance days into multi-day events, and evaluates group SHAP attributions for each episode.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def detect_events(self, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Detects extreme pollution events and exports event catalog & attributions."""
        logger.info("Detecting Extreme Pollution Events (PM2.5 >= 90th percentile)...")

        p90_threshold = float(df["pm25"].quantile(0.90))
        logger.info(f"Using established 90th Percentile PM2.5 Threshold: {p90_threshold:.2f} µg/m³.")

        exceedance_mask = df["pm25"] >= p90_threshold
        dates_dt = pd.to_datetime(df["date"])

        events = []
        in_event = False
        current_event_dates = []

        for idx, row in df.iterrows():
            is_high = row["pm25"] >= p90_threshold
            if is_high:
                current_event_dates.append(idx)
                in_event = True
            else:
                if in_event:
                    events.append(current_event_dates)
                    current_event_dates = []
                    in_event = False

        if in_event and current_event_dates:
            events.append(current_event_dates)

        logger.info(f"Identified {len(events)} distinct high-pollution episodes (total exceedance days: {exceedance_mask.sum()}).")

        fire_col = "fire_hotspot_count_lag_1d" if "fire_hotspot_count_lag_1d" in df.columns else "fire_hotspot_count_roll_mean_7d"
        wind_col = "wind_speed_kmh_lag_1d" if "wind_speed_kmh_lag_1d" in df.columns else "wind_speed_kmh_roll_mean_7d"

        event_catalog_rows = []
        event_attr_rows = []

        ordered_groups = ["pm25_persistence", "meteorology", "wind_ventilation", "biomass_burning", "calendar_seasonal"]

        for event_id, idx_list in enumerate(events, start=1):
            e_df = df.loc[idx_list]
            e_shap_df = group_shap_df.loc[idx_list]

            start_date = e_df["date"].iloc[0]
            end_date = e_df["date"].iloc[-1]

            peak_idx = e_df["pm25"].idxmax()
            peak_date = df.loc[peak_idx, "date"]
            peak_pm25 = float(df.loc[peak_idx, "pm25"])
            mean_pm25 = float(e_df["pm25"].mean())

            mean_biomass_shap = float(e_shap_df["biomass_burning_shap"].mean())
            mean_met_shap = float(e_shap_df["meteorology_shap"].mean())
            mean_wind_shap = float(e_shap_df["wind_ventilation_shap"].mean())
            mean_pers_shap = float(e_shap_df["pm25_persistence_shap"].mean())

            group_means = {grp: float(e_shap_df[f"{grp}_shap"].mean()) for grp in ordered_groups}
            dominant_group = max(group_means, key=group_means.get)

            mean_fire = float(e_df[fire_col].mean()) if fire_col in e_df.columns else 0.0
            mean_wind = float(e_df[wind_col].mean()) if wind_col in e_df.columns else 0.0

            event_catalog_rows.append({
                "event_id": f"EVT_{event_id:03d}",
                "event_start": start_date,
                "event_end": end_date,
                "duration_days": len(idx_list),
                "peak_date": peak_date,
                "peak_pm25": peak_pm25,
                "mean_pm25": mean_pm25,
                "biomass_burning_shap": mean_biomass_shap,
                "meteorology_shap": mean_met_shap,
                "wind_ventilation_shap": mean_wind_shap,
                "pm25_persistence_shap": mean_pers_shap,
                "dominant_attribution_group": dominant_group,
                "mean_fire_hotspots": mean_fire,
                "mean_wind_speed_kmh": mean_wind
            })

            for idx in idx_list:
                event_attr_rows.append({
                    "event_id": f"EVT_{event_id:03d}",
                    "date": df.loc[idx, "date"],
                    "pm25": float(df.loc[idx, "pm25"]),
                    "predicted_pm25": float(group_shap_df.loc[idx, "predicted_pm25"]),
                    "biomass_burning_shap": float(group_shap_df.loc[idx, "biomass_burning_shap"]),
                    "meteorology_shap": float(group_shap_df.loc[idx, "meteorology_shap"]),
                    "wind_ventilation_shap": float(group_shap_df.loc[idx, "wind_ventilation_shap"]),
                    "pm25_persistence_shap": float(group_shap_df.loc[idx, "pm25_persistence_shap"]),
                    "calendar_seasonal_shap": float(group_shap_df.loc[idx, "calendar_seasonal_shap"])
                })

        catalog_df = pd.DataFrame(event_catalog_rows)
        attr_df = pd.DataFrame(event_attr_rows)

        catalog_df.to_csv(self.exp_dir / "event_catalog.csv", index=False)
        attr_df.to_csv(self.exp_dir / "event_attributions.csv", index=False)

        logger.info(f"Event catalog created with {len(catalog_df)} episodes. Exported to {self.exp_dir / 'event_catalog.csv'}.")

        return catalog_df, attr_df


if __name__ == "__main__":
    detector = EventDetectionPhase4C()
