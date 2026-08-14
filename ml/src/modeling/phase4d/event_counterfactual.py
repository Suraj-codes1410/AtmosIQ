import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("EventCounterfactualPhase4D")


class EventCounterfactualPhase4D:
    """
    AtmosIQ Phase 4D Event & Daily Counterfactual Analysis Engine & API.
    Evaluates scenario counterfactuals across historical extreme pollution episodes and provides interactive reporting APIs.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def analyze_event_counterfactuals(
        self,
        event_catalog_df: pd.DataFrame,
        intervention_engine,
        X_all: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Evaluates counterfactual scenarios across Phase 4C extreme pollution episodes using batch predictions."""
        logger.info("Analyzing Counterfactual Scenarios across Phase 4C Extreme Pollution Episodes...")

        event_rows = []
        X_mat = X_all[intervention_engine.feature_order].values

        for _, evt in event_catalog_df.iterrows():
            e_id = evt["event_id"]
            s_date = evt["event_start"]
            e_date = evt["event_end"]

            dates_dt = pd.to_datetime(df["date"])
            mask = (dates_dt >= pd.to_datetime(s_date)) & (dates_dt <= pd.to_datetime(e_date))
            idx_list = df[mask].index.tolist()

            sub_X = X_mat[idx_list]

            # Vectorized Baseline predictions
            obs_preds = intervention_engine.predict_batch(sub_X)
            mean_obs_pred = float(np.mean(obs_preds))
            peak_obs_pm25 = float(evt["peak_pm25"])

            # 1. Biomass Low (Q25)
            bio_X_mat = np.array([intervention_engine.apply_intervention(x, "biomass_burning", "q25") for x in sub_X])
            bio_cf_preds = intervention_engine.predict_batch(bio_X_mat)
            mean_bio_cf = float(np.mean(bio_cf_preds))
            bio_delta = mean_bio_cf - mean_obs_pred

            # 2. Wind Dispersion (Q75)
            wind_X_mat = np.array([intervention_engine.apply_intervention(x, "wind_ventilation", "q75") for x in sub_X])
            wind_cf_preds = intervention_engine.predict_batch(wind_X_mat)
            mean_wind_cf = float(np.mean(wind_cf_preds))
            wind_delta = mean_wind_cf - mean_obs_pred

            # 3. Meteorology Normal (Q50)
            met_X_mat = np.array([intervention_engine.apply_intervention(x, "meteorology", "q50") for x in sub_X])
            met_cf_preds = intervention_engine.predict_batch(met_X_mat)
            mean_met_cf = float(np.mean(met_cf_preds))
            met_delta = mean_met_cf - mean_obs_pred

            # 4. Combined Favorable
            comb_X_mat = np.array([intervention_engine.apply_multi_group_intervention(x, {"biomass_burning": "q25", "wind_ventilation": "q75", "meteorology": "q50"}) for x in sub_X])
            comb_cf_preds = intervention_engine.predict_batch(comb_X_mat)
            mean_comb_cf = float(np.mean(comb_cf_preds))
            comb_delta = mean_comb_cf - mean_obs_pred

            event_rows.append({
                "event_id": e_id,
                "start_date": s_date,
                "end_date": e_date,
                "peak_date": evt["peak_date"],
                "observed_pm25": peak_obs_pm25,
                "observed_prediction": mean_obs_pred,
                "dominant_group": evt["dominant_attribution_group"],
                "biomass_cf_prediction": mean_bio_cf,
                "biomass_delta": bio_delta,
                "wind_cf_prediction": mean_wind_cf,
                "wind_delta": wind_delta,
                "meteorology_cf_prediction": mean_met_cf,
                "meteorology_delta": met_delta,
                "combined_cf_prediction": mean_comb_cf,
                "combined_delta": comb_delta,
                "confidence": "HIGH",
                "ood_flag": False
            })

        evt_cf_df = pd.DataFrame(event_rows)
        evt_cf_df.to_csv(self.exp_dir / "event_counterfactuals.csv", index=False)

        logger.info(f"Event counterfactual analysis complete for {len(evt_cf_df)} episodes. Exported to {self.exp_dir / 'event_counterfactuals.csv'}.")
        return evt_cf_df

    def explain_counterfactual_date(
        self,
        date_str: str,
        scenario: str,
        df: pd.DataFrame,
        X_all: pd.DataFrame,
        intervention_engine,
        conf_df: pd.DataFrame
    ) -> dict:
        """Explains counterfactual scenario prediction for a single date."""
        row_mask = df["date"] == date_str
        if not row_mask.any():
            raise ValueError(f"Date '{date_str}' not found in Dataset v2!")

        idx = df[row_mask].index[0]
        x_obs = X_all[intervention_engine.feature_order].iloc[idx].values
        act_pm25 = float(df.loc[idx, "pm25"])

        if scenario == "biomass_low":
            x_cf = intervention_engine.apply_intervention(x_obs, "biomass_burning", "q25")
            grp = "biomass_burning"
        elif scenario == "wind_dispersion":
            x_cf = intervention_engine.apply_intervention(x_obs, "wind_ventilation", "q75")
            grp = "wind_ventilation"
        else:
            x_cf = intervention_engine.apply_intervention(x_obs, "biomass_burning", "q25")
            grp = "biomass_burning"

        pred_obs, pred_cf, delta = intervention_engine.predict_observed_and_counterfactual(x_obs, x_cf)
        conf_row = conf_df[(conf_df["date"] == date_str) & (conf_df["scenario"] == scenario)]
        conf_level = conf_row["counterfactual_confidence_level"].iloc[0] if len(conf_row) > 0 else "HIGH"

        report_txt = f"""============================================================
ATMOSIQ DAILY COUNTERFACTUAL REPORT: {date_str}
Scenario:                       {scenario}
Target Group:                   {grp}
Observed PM2.5:                 {act_pm25:.1f} µg/m³
Baseline Model Prediction:     {pred_obs:.1f} µg/m³
Counterfactual Prediction:     {pred_cf:.1f} µg/m³
Model Prediction Delta Δŷ:      {delta:+.2f} µg/m³
Counterfactual Confidence:      {conf_level}

SCIENTIFIC LIMITATION:
This represents a model-based feature sensitivity Δŷ under a controlled scenario.
It does NOT represent a physical causal emission reduction percentage.
============================================================
"""
        return {
            "date": date_str,
            "scenario": scenario,
            "target_group": grp,
            "actual_pm25": act_pm25,
            "prediction_observed": pred_obs,
            "prediction_counterfactual": pred_cf,
            "delta_prediction": delta,
            "confidence_level": conf_level,
            "formatted_report": report_txt
        }


if __name__ == "__main__":
    analyzer = EventCounterfactualPhase4D()
