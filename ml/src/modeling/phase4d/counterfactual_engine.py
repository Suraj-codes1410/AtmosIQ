import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger

logger = setup_logger("CounterfactualEnginePhase4D")


class CounterfactualSimulationEnginePhase4D:
    """
    AtmosIQ Phase 4D Master Counterfactual Simulation Engine.
    Executes single-group, multi-group, and scenario counterfactual simulations across all 1,827 observations in Dataset v2.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def run_simulations(self, intervention_engine, scenarios_dict: dict, X_all: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs all counterfactual scenarios across Dataset v2 using vectorized batch predictions."""
        logger.info("Executing Master Counterfactual Simulations across 1,827 daily observations...")

        intervention_engine.fit_reference_quantiles(X_all)

        X_mat = X_all[intervention_engine.feature_order].values
        dates = df["date"].tolist()
        pm25_vals = df["pm25"].values

        p90_threshold = float(df["pm25"].quantile(0.90))

        # Vectorized baseline prediction
        preds_obs = intervention_engine.predict_batch(X_mat)

        results_rows = []

        for scen_name, scen_info in scenarios_dict.items():
            target_grp = scen_info["group"]
            logger.info(f"Simulating scenario: {scen_name} ({target_grp})...")

            # Build batch counterfactual matrix
            X_cf_mat = np.zeros_like(X_mat)
            for i in range(len(dates)):
                x_obs = X_mat[i]
                if target_grp == "multi_group":
                    grp_q_map = {}
                    if "groups" in scen_info:
                        for g in scen_info["groups"]:
                            grp_q_map[g] = "q25" if g == "biomass_burning" else ("q75" if g == "wind_ventilation" else "q50")
                    X_cf_mat[i] = intervention_engine.apply_multi_group_intervention(x_obs, grp_q_map)
                else:
                    q_key = scen_info.get("quantile_key", "q25" if "low" in scen_name or "stagnant" in scen_name else ("q75" if "high" in scen_name or "dispersion" in scen_name else "q50"))
                    X_cf_mat[i] = intervention_engine.apply_intervention(x_obs, target_grp, q_key)

            # Bulk predict
            preds_cf = intervention_engine.predict_batch(X_cf_mat)
            deltas = preds_cf - preds_obs

            for i in range(len(dates)):
                results_rows.append({
                    "date": dates[i],
                    "actual_pm25": float(pm25_vals[i]),
                    "prediction_observed": float(preds_obs[i]),
                    "scenario": scen_name,
                    "target_group": target_grp,
                    "prediction_counterfactual": float(preds_cf[i]),
                    "delta_prediction": float(deltas[i]),
                    "is_extreme_day": pm25_vals[i] >= p90_threshold
                })

        cf_results_df = pd.DataFrame(results_rows)
        cf_results_df.to_csv(self.exp_dir / "counterfactual_results.csv", index=False)

        # Build Group Counterfactual Summary Table
        summary_rows = []
        for scen_name in scenarios_dict.keys():
            scen_df = cf_results_df[cf_results_df["scenario"] == scen_name]
            deltas = scen_df["delta_prediction"].values

            normal_deltas = scen_df.loc[~scen_df["is_extreme_day"], "delta_prediction"].values
            extreme_deltas = scen_df.loc[scen_df["is_extreme_day"], "delta_prediction"].values

            summary_rows.append({
                "scenario": scen_name,
                "target_group": scenarios_dict[scen_name]["group"],
                "mean_delta_all": float(np.mean(deltas)),
                "median_delta_all": float(np.median(deltas)),
                "std_delta_all": float(np.std(deltas)),
                "iqr_delta_all": float(np.percentile(deltas, 75) - np.percentile(deltas, 25)),
                "p05_delta_all": float(np.percentile(deltas, 5)),
                "p25_delta_all": float(np.percentile(deltas, 25)),
                "p75_delta_all": float(np.percentile(deltas, 75)),
                "p95_delta_all": float(np.percentile(deltas, 95)),
                "mean_delta_normal_days": float(np.mean(normal_deltas)),
                "mean_delta_extreme_days": float(np.mean(extreme_deltas)) if len(extreme_deltas) > 0 else 0.0
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.exp_dir / "group_counterfactual_summary.csv", index=False)

        logger.info(f"Counterfactual simulation complete. Summary exported to {self.exp_dir / 'group_counterfactual_summary.csv'}.")
        return cf_results_df, summary_df


if __name__ == "__main__":
    sim_engine = CounterfactualSimulationEnginePhase4D()
