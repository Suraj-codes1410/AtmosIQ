import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("ConfidencePhase4D")


class ConfidenceEvaluatorPhase4D:
    """
    AtmosIQ Phase 4D Counterfactual Confidence & Evidence Scorer.
    Evaluates 4-tier confidence rating (HIGH, MODERATE, LOW, INVALID) combining plausibility, OOD status, SHAP consistency, and Phase 4C evidence scores.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_confidence(self, cf_results_df: pd.DataFrame, plausibility_df: pd.DataFrame, ood_df: pd.DataFrame, consistency_df: pd.DataFrame, p4c_conf_df: pd.DataFrame) -> pd.DataFrame:
        """Assigns 4-tier confidence scores for all counterfactual predictions."""
        logger.info("Evaluating 4-tier counterfactual confidence scores...")

        rows = []
        for i in range(len(cf_results_df)):
            cf_row = cf_results_df.iloc[i]
            dt = cf_row["date"]
            scen = cf_row["scenario"]

            p_pass = plausibility_df.loc[i, "overall_plausibility_pass"] if i in plausibility_df.index else True
            ood_flag = ood_df.loc[i, "ood_flag"] if i in ood_df.index else False
            consist = consistency_df.loc[i, "directional_consistency"] if i in consistency_df.index else True

            # Match Phase 4C evidence score
            p4c_match = p4c_conf_df[p4c_conf_df["date"] == dt]
            e_score = int(p4c_match["evidence_score"].iloc[0]) if len(p4c_match) > 0 else 1

            if not p_pass:
                conf = "INVALID"
            elif ood_flag or not consist:
                conf = "LOW"
            elif e_score >= 2:
                conf = "HIGH"
            else:
                conf = "MODERATE"

            rows.append({
                "date": dt,
                "scenario": scen,
                "plausibility_pass": p_pass,
                "ood_flag": ood_flag,
                "directional_consistency": consist,
                "phase4c_evidence_score": e_score,
                "counterfactual_confidence_level": conf
            })

        conf_df = pd.DataFrame(rows)
        conf_df.to_csv(self.exp_dir / "confidence_scores.csv", index=False)

        high_pct = float(np.mean(conf_df["counterfactual_confidence_level"] == "HIGH")) * 100
        mod_pct = float(np.mean(conf_df["counterfactual_confidence_level"] == "MODERATE")) * 100
        logger.info(f"Counterfactual Confidence Evaluation complete -> HIGH: {high_pct:.1f}%, MODERATE: {mod_pct:.1f}%.")

        return conf_df


if __name__ == "__main__":
    evaluator = ConfidenceEvaluatorPhase4D()
