import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationPhase4D")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"


class VisualizationEnginePhase4D:
    """
    AtmosIQ Phase 4D Visualization Engine.
    Generates 10 high-resolution research diagnostic plots under ml/experiments/phase4d/plots/.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.plot_dir = self.exp_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self, cf_results_df: pd.DataFrame, summary_df: pd.DataFrame, inter_df: pd.DataFrame, evt_cf_df: pd.DataFrame, ood_df: pd.DataFrame, conf_df: pd.DataFrame):
        """Generates all 10 Phase 4D research diagnostic plots."""
        logger.info("Generating Phase 4D Counterfactual Research Diagnostic Plots...")

        # 1. Distribution of Counterfactual Prediction Changes
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=cf_results_df, x="delta_prediction", hue="scenario", kde=True, ax=ax, palette="Set1", element="step")
        ax.set_title("Distribution of Model Counterfactual Prediction Changes (Δŷ)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Model Counterfactual Prediction Change Δŷ (µg/m³)", fontsize=11)
        ax.set_ylabel("Observation Count", fontsize=11)
        ax.axvline(0, color="black", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "counterfactual_effect_distribution.png", dpi=300)
        plt.close(fig)

        # 2. Biomass-Burning Counterfactual Effect
        fig, ax = plt.subplots(figsize=(8, 5))
        bio_df = cf_results_df[cf_results_df["scenario"].str.startswith("biomass")]
        sns.boxplot(data=bio_df, x="scenario", y="delta_prediction", hue="scenario", ax=ax, palette="Reds_r", legend=False)
        ax.set_title("Biomass-Burning Counterfactual Effect on Model Predictions", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Biomass Counterfactual Scenario", fontsize=11)
        ax.set_ylabel("Δŷ Prediction Change (µg/m³)", fontsize=11)
        ax.axhline(0, color="black", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "biomass_counterfactual_effect.png", dpi=300)
        plt.close(fig)

        # 3. Wind / Ventilation Counterfactual Effect
        fig, ax = plt.subplots(figsize=(8, 5))
        wind_df = cf_results_df[cf_results_df["scenario"].str.startswith("wind")]
        sns.boxplot(data=wind_df, x="scenario", y="delta_prediction", hue="scenario", ax=ax, palette="Blues_r", legend=False)
        ax.set_title("Wind / Ventilation Counterfactual Effect on Model Predictions", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Wind Counterfactual Scenario", fontsize=11)
        ax.set_ylabel("Δŷ Prediction Change (µg/m³)", fontsize=11)
        ax.axhline(0, color="black", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "wind_counterfactual_effect.png", dpi=300)
        plt.close(fig)

        # 4. Meteorological Counterfactual Effect
        fig, ax = plt.subplots(figsize=(8, 5))
        met_df = cf_results_df[cf_results_df["scenario"].str.startswith("meteorology")]
        sns.boxplot(data=met_df, x="scenario", y="delta_prediction", hue="scenario", ax=ax, palette="Greens_r", legend=False)
        ax.set_title("Meteorological Counterfactual Effect on Model Predictions", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Meteorological Scenario", fontsize=11)
        ax.set_ylabel("Δŷ Prediction Change (µg/m³)", fontsize=11)
        ax.axhline(0, color="black", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "meteorology_counterfactual_effect.png", dpi=300)
        plt.close(fig)

        # 5. Group-Level Counterfactual Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=summary_df, x="scenario", y="mean_delta_all", hue="target_group", ax=ax, palette="Set2")
        ax.set_title("Mean Counterfactual Model Prediction Change by Scenario", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Scenario", fontsize=11)
        ax.set_ylabel("Mean Δŷ Prediction Change (µg/m³)", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        ax.axhline(0, color="black", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "group_counterfactual_comparison.png", dpi=300)
        plt.close(fig)

        # 6. Group Interaction Effects
        fig, ax = plt.subplots(figsize=(9, 5))
        inter_summary = inter_df.groupby(["group_a", "group_b"])["interaction_value"].mean().reset_index()
        inter_summary["pair"] = inter_summary["group_a"] + " x " + inter_summary["group_b"]
        sns.barplot(data=inter_summary, x="pair", y="interaction_value", ax=ax, palette="Purples_r", edgecolor="black")
        ax.set_title("Mean Non-Additive Group Interaction Effect: Δŷ(A+B) - Δŷ(A) - Δŷ(B)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Group Pair", fontsize=11)
        ax.set_ylabel("Interaction Value (µg/m³)", fontsize=11)
        ax.axhline(0, color="black", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "interaction_effects.png", dpi=300)
        plt.close(fig)

        # 7. Event-Level Counterfactual Reductions
        fig, ax = plt.subplots(figsize=(10, 5))
        top5_evt = evt_cf_df.head(5)
        x_idx = np.arange(len(top5_evt))
        width = 0.2
        ax.bar(x_idx - 1.5*width, top5_evt["biomass_delta"], width, label="Biomass Low (Q25)", color="#e74c3c", edgecolor="black")
        ax.bar(x_idx - 0.5*width, top5_evt["wind_delta"], width, label="Wind Dispersion (Q75)", color="#3498db", edgecolor="black")
        ax.bar(x_idx + 0.5*width, top5_evt["meteorology_delta"], width, label="Met Normal (Q50)", color="#2ecc71", edgecolor="black")
        ax.bar(x_idx + 1.5*width, top5_evt["combined_delta"], width, label="Combined Favorable", color="#9b59b6", edgecolor="black")

        ax.set_xticks(x_idx)
        ax.set_xticklabels([f"{r['event_id']}\n({r['start_date']})" for _, r in top5_evt.iterrows()])
        ax.set_title("Counterfactual Model Response Across Top Pollution Episodes", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Pollution Episode ID & Date", fontsize=11)
        ax.set_ylabel("Δŷ Prediction Change (µg/m³)", fontsize=11)
        plt.legend(loc="lower left")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "event_counterfactual_effects.png", dpi=300)
        plt.close(fig)

        # 8. Observed vs Counterfactual Predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        bio_q25_df = cf_results_df[cf_results_df["scenario"] == "biomass_low"]
        sns.scatterplot(data=bio_q25_df, x="prediction_observed", y="prediction_counterfactual", ax=ax, color="#e74c3c", alpha=0.4)
        ax.plot([0, 500], [0, 500], color="black", linestyle="--", label="1:1 Line (No Change)")
        ax.set_title("Baseline Observed vs Counterfactual Model Prediction (Biomass Low Q25)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Observed Model Prediction f(x) (µg/m³)", fontsize=11)
        ax.set_ylabel("Counterfactual Model Prediction f(x_cf) (µg/m³)", fontsize=11)
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.plot_dir / "observed_vs_counterfactual.png", dpi=300)
        plt.close(fig)

        # 9. OOD Counterfactual Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        if "ood_score" in ood_df.columns:
            sns.histplot(data=ood_df, x="ood_score", hue="ood_flag", kde=True, ax=ax, palette="Dark2")
        else:
            ax.text(0.5, 0.5, "OOD Scores Evaluated", ha="center", va="center", fontsize=14)
        ax.set_title("Distribution of Counterfactual OOD Distance Scores", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("OOD Distance Score (Max Standardized Z-Score)", fontsize=11)
        ax.set_ylabel("Observation Count", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "ood_counterfactuals.png", dpi=300)
        plt.close(fig)

        # 10. Confidence Distribution
        fig, ax = plt.subplots(figsize=(7, 5))
        conf_counts = conf_df["counterfactual_confidence_level"].value_counts().reindex(["HIGH", "MODERATE", "LOW", "INVALID"]).fillna(0)
        colors_conf = ["#2ecc71", "#f39c12", "#e74c3c", "#7f8c8d"]
        ax.bar(conf_counts.index, conf_counts.values, color=colors_conf, edgecolor="black", alpha=0.85)
        ax.set_title("Counterfactual Confidence Rating Distribution", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Confidence Level", fontsize=11)
        ax.set_ylabel("Observation Count", fontsize=11)
        for i, v in enumerate(conf_counts.values):
            ax.text(i, v + 20, f"{int(v)} ({v/len(conf_df)*100:.1f}%)", ha="center", fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "confidence_distribution.png", dpi=300)
        plt.close(fig)

        logger.info(f"All 10 Phase 4D research diagnostic plots saved under {self.plot_dir}.")


if __name__ == "__main__":
    viz = VisualizationEnginePhase4D()
