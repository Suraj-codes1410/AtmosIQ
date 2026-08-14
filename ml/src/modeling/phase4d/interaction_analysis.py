import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("InteractionAnalysisPhase4D")


class InteractionAnalysisPhase4D:
    """
    AtmosIQ Phase 4D Multi-Group Interaction Analysis Engine.
    Evaluates non-additive model interaction effects: interaction(A,B) = effect(A+B) - effect(A) - effect(B).
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def compute_interactions(self, intervention_engine, X_all: pd.DataFrame, dates: list) -> pd.DataFrame:
        """Computes multi-group non-additive interactions using vectorized predictions."""
        logger.info("Computing Multi-Group Non-Additive Counterfactual Interaction Effects...")

        pairs = [
            ("biomass_burning", "wind_ventilation", "q25", "q75"),
            ("biomass_burning", "meteorology", "q25", "q50"),
            ("wind_ventilation", "meteorology", "q75", "q50")
        ]

        X_mat = X_all[intervention_engine.feature_order].values
        preds_obs = intervention_engine.predict_batch(X_mat)

        rows = []

        for grp_a, grp_b, q_a, q_b in pairs:
            # Vectorized Matrix A
            X_a_mat = np.zeros_like(X_mat)
            for i in range(len(dates)):
                X_a_mat[i] = intervention_engine.apply_intervention(X_mat[i], grp_a, q_a)
            preds_a = intervention_engine.predict_batch(X_a_mat)
            effects_a = preds_a - preds_obs

            # Vectorized Matrix B
            X_b_mat = np.zeros_like(X_mat)
            for i in range(len(dates)):
                X_b_mat[i] = intervention_engine.apply_intervention(X_mat[i], grp_b, q_b)
            preds_b = intervention_engine.predict_batch(X_b_mat)
            effects_b = preds_b - preds_obs

            # Vectorized Matrix A+B
            X_ab_mat = np.zeros_like(X_mat)
            for i in range(len(dates)):
                X_ab_mat[i] = intervention_engine.apply_multi_group_intervention(X_mat[i], {grp_a: q_a, grp_b: q_b})
            preds_ab = intervention_engine.predict_batch(X_ab_mat)
            effects_ab = preds_ab - preds_obs

            interactions = effects_ab - effects_a - effects_b

            for i in range(len(dates)):
                rows.append({
                    "date": dates[i],
                    "group_a": grp_a,
                    "group_b": grp_b,
                    "effect_a": float(effects_a[i]),
                    "effect_b": float(effects_b[i]),
                    "effect_combined_ab": float(effects_ab[i]),
                    "interaction_value": float(interactions[i]),
                    "non_additive_flag": bool(abs(interactions[i]) > 2.0)
                })

        df_res = pd.DataFrame(rows)
        df_res.to_csv(self.exp_dir / "interaction_analysis.csv", index=False)

        mean_inter_bw = float(np.mean(df_res.loc[(df_res["group_a"] == "biomass_burning") & (df_res["group_b"] == "wind_ventilation"), "interaction_value"]))
        logger.info(f"Interaction analysis complete. Mean Biomass x Wind Interaction: {mean_inter_bw:.2f} µg/m³.")

        return df_res


if __name__ == "__main__":
    analyzer = InteractionAnalysisPhase4D()
