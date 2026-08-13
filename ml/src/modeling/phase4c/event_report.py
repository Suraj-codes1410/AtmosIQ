import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("EventReportPhase4C")


class EventReportPhase4C:
    """
    AtmosIQ Phase 4C Pollution Event & Date Attribution Reporting API.
    Provides explain_event(start_date, end_date) and explain_date(date) generating formatted environmental attribution reports.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)

    def explain_date(self, date_str: str, df: pd.DataFrame, group_shap_df: pd.DataFrame, conf_df: pd.DataFrame) -> dict:
        """Explains environmental attribution for a single date."""
        row_mask = df["date"] == date_str
        if not row_mask.any():
            raise ValueError(f"Date '{date_str}' not found in Dataset v2!")

        idx = df[row_mask].index[0]
        act_pm25 = float(df.loc[idx, "pm25"])
        pred_pm25 = float(group_shap_df.loc[idx, "predicted_pm25"])
        base_val = float(group_shap_df.loc[idx, "base_value"])
        conf_row = conf_df[conf_df["date"] == date_str].iloc[0]

        grp_cols = ["pm25_persistence_shap", "meteorology_shap", "wind_ventilation_shap", "biomass_burning_shap", "calendar_seasonal_shap"]
        grp_vals = {c.replace("_shap", ""): float(group_shap_df.loc[idx, c]) for c in grp_cols}

        ranked_groups = sorted(grp_vals.items(), key=lambda x: abs(x[1]), reverse=True)

        report_txt = f"""============================================================
ATMOSIQ POLLUTION DATE REPORT: {date_str}
============================================================
Actual PM2.5:                    {act_pm25:.1f} µg/m³
Model Predicted PM2.5:          {pred_pm25:.1f} µg/m³
Expected Base Value:            {base_val:.1f} µg/m³
Attribution Confidence Level:   {conf_row['confidence_level']} (Score: {conf_row['evidence_score']}/3)

MODEL ATTRIBUTION BY GROUP:
1. {ranked_groups[0][0]}: {ranked_groups[0][1]:+.2f} µg/m³
2. {ranked_groups[1][0]}: {ranked_groups[1][1]:+.2f} µg/m³
3. {ranked_groups[2][0]}: {ranked_groups[2][1]:+.2f} µg/m³
4. {ranked_groups[3][0]}: {ranked_groups[3][1]:+.2f} µg/m³
5. {ranked_groups[4][0]}: {ranked_groups[4][1]:+.2f} µg/m³

SCIENTIFIC LIMITATION:
These values explain the predictive model f(x) and do not represent physical causal emission-source percentages.
============================================================
"""
        return {
            "date": date_str,
            "actual_pm25": act_pm25,
            "predicted_pm25": pred_pm25,
            "base_value": base_val,
            "confidence_level": conf_row["confidence_level"],
            "evidence_score": int(conf_row["evidence_score"]),
            "ranked_groups": ranked_groups,
            "formatted_report": report_txt
        }

    def explain_event(self, start_date: str, end_date: str, df: pd.DataFrame, group_shap_df: pd.DataFrame, conf_df: pd.DataFrame) -> dict:
        """Explains environmental attribution for a multi-day pollution episode."""
        dates_dt = pd.to_datetime(df["date"])
        mask = (dates_dt >= pd.to_datetime(start_date)) & (dates_dt <= pd.to_datetime(end_date))

        if mask.sum() == 0:
            raise ValueError(f"No observations found between {start_date} and {end_date}!")

        sub_df = df[mask]
        sub_group_df = group_shap_df[mask]
        sub_conf_df = conf_df[mask]

        peak_idx = sub_df["pm25"].idxmax()
        peak_date = sub_df.loc[peak_idx, "date"]
        peak_pm25 = float(sub_df.loc[peak_idx, "pm25"])
        mean_pm25 = float(sub_df["pm25"].mean())

        grp_cols = ["pm25_persistence_shap", "meteorology_shap", "wind_ventilation_shap", "biomass_burning_shap", "calendar_seasonal_shap"]
        mean_grp_vals = {c.replace("_shap", ""): float(sub_group_df[c].mean()) for c in grp_cols}
        ranked_groups = sorted(mean_grp_vals.items(), key=lambda x: abs(x[1]), reverse=True)

        avg_score = float(sub_conf_df["evidence_score"].mean())
        if avg_score >= 1.5:
            conf_level = "High" if avg_score >= 2.2 else "Moderate"
        else:
            conf_level = "Low"

        report_txt = f"""============================================================
ATMOSIQ POLLUTION EPISODE REPORT
Period:                         {start_date} to {end_date} ({mask.sum()} days)
Peak PM2.5:                     {peak_pm25:.1f} µg/m³ (on {peak_date})
Episode Mean PM2.5:            {mean_pm25:.1f} µg/m³
Average Environmental Support:  {conf_level} (Mean Score: {avg_score:.2f}/3)

DOMINANT MODEL ATTRIBUTIONS (EPISODE AVERAGE):
1. {ranked_groups[0][0]}: {ranked_groups[0][1]:+.2f} µg/m³
2. {ranked_groups[1][0]}: {ranked_groups[1][1]:+.2f} µg/m³
3. {ranked_groups[2][0]}: {ranked_groups[2][1]:+.2f} µg/m³
4. {ranked_groups[3][0]}: {ranked_groups[3][1]:+.2f} µg/m³
5. {ranked_groups[4][0]}: {ranked_groups[4][1]:+.2f} µg/m³

SCIENTIFIC LIMITATION:
These values explain the predictive model f(x) and do not represent physical causal emission-source percentages.
============================================================
"""
        return {
            "start_date": start_date,
            "end_date": end_date,
            "duration_days": int(mask.sum()),
            "peak_pm25": peak_pm25,
            "peak_date": peak_date,
            "mean_pm25": mean_pm25,
            "confidence_level": conf_level,
            "average_evidence_score": avg_score,
            "ranked_groups": ranked_groups,
            "formatted_report": report_txt
        }


if __name__ == "__main__":
    reporter = EventReportPhase4C()
