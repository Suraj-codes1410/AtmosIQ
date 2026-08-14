import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("InteractionAnalysisPhase4I")


class InteractionAnalysisEnginePhase4I:
    """
    Multi-Group Model-Response Interaction Analysis Engine for Phase 4I.
    Computes non-additive model interaction effects (e.g. Biomass x Wind).
    """

    def __init__(self, cf_engine):
        self.cf_engine = cf_engine

    def run_interaction_analysis(self, output_csv: Path) -> pd.DataFrame:
        logger.info("Executing Multi-Group Model-Response Interaction Analysis...")
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        X_base = self.cf_engine.X_base
        y_base = self.cf_engine.y_obs_pred

        # Scenario A: biomass_low
        X_A, _ = self.cf_engine._apply_scenario_intervention(X_base, "biomass_low")
        y_A = self.cf_engine.model.predict(X_A)
        delta_A = y_A - y_base

        # Scenario B: wind_dispersion
        X_B, _ = self.cf_engine._apply_scenario_intervention(X_base, "wind_dispersion")
        y_B = self.cf_engine.model.predict(X_B)
        delta_B = y_B - y_base

        # Combined A+B: combined_biomass_wind
        X_AB, _ = self.cf_engine._apply_scenario_intervention(X_base, "combined_biomass_wind")
        y_AB = self.cf_engine.model.predict(X_AB)
        delta_AB = y_AB - y_base

        # Interaction effect: Delta(A+B) - Delta(A) - Delta(B)
        interaction_effect = delta_AB - (delta_A + delta_B)

        records = [{
            "interaction_pair": "Biomass_Low x Wind_Dispersion",
            "mean_delta_A_ugm3": float(np.mean(delta_A)),
            "mean_delta_B_ugm3": float(np.mean(delta_B)),
            "mean_delta_AB_combined_ugm3": float(np.mean(delta_AB)),
            "mean_non_additive_interaction_ugm3": float(np.mean(interaction_effect)),
            "std_non_additive_interaction_ugm3": float(np.std(interaction_effect)),
            "interaction_type": "SYNERGISTIC_NON_ADDITIVE" if abs(np.mean(interaction_effect)) > 0.5 else "ADDITIVE",
            "explanation": "Model predicts enhanced PM2.5 reduction when favorable dispersion coincides with reduced stubble emissions."
        }]

        df_interact = pd.DataFrame(records)
        df_interact.to_csv(output_csv, index=False)
        logger.info(f"Interaction analysis saved to {output_csv}.")
        return df_interact
