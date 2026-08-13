import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationPhase4B")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"


class VisualizationEnginePhase4B:
    """
    AtmosIQ Phase 4B Visualization Engine.
    Generates high-resolution diagnostic plots under ml/experiments/phase4b/plots/.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4b"):
        self.exp_dir = Path(exp_dir)
        self.plot_dir = self.exp_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self, feat_imp_df: pd.DataFrame, grp_imp_df: pd.DataFrame, seasonal_df: pd.DataFrame, high_analysis_df: pd.DataFrame, shap_matrix: np.ndarray, X_all: pd.DataFrame, feature_order: list, rep_dates_info: dict, base_value: float):
        """Generates all global and local diagnostic plots."""
        logger.info("Generating Phase 4B SHAP diagnostic plots...")

        # 1. Global Feature Importance (Top 20)
        fig, ax = plt.subplots(figsize=(10, 7))
        top20 = feat_imp_df.head(20).sort_values("mean_abs_shap", ascending=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top20)))
        ax.barh(top20["feature_name"], top20["mean_abs_shap"], color=colors, edgecolor="black", alpha=0.85)
        ax.set_title("AtmosIQ Phase 4B: Top 20 Global Feature SHAP Importance", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Mean |SHAP Value| (µg/m³)", fontsize=12)
        ax.set_ylabel("Feature Name", fontsize=12)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "global_feature_importance.png", dpi=300)
        plt.close(fig)

        # 2. SHAP Beeswarm Summary Plot
        try:
            plt.figure(figsize=(11, 7))
            shap.summary_plot(shap_matrix, X_all, feature_names=feature_order, max_display=20, show=False)
            plt.title("AtmosIQ Phase 4B: TreeSHAP Beeswarm Summary Plot", fontsize=14, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.savefig(self.plot_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate beeswarm summary plot: {e}")

        # 3. Global Group Importance
        fig, ax = plt.subplots(figsize=(9, 5))
        sorted_grp = grp_imp_df.sort_values("mean_abs_shap", ascending=True)
        colors_grp = sns.color_palette("deep", len(sorted_grp))
        ax.barh(sorted_grp["attribution_group"], sorted_grp["mean_abs_shap"], color=colors_grp, edgecolor="black", alpha=0.85)
        ax.set_title("AtmosIQ Phase 4B: Global Environmental Group SHAP Importance", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Mean |SHAP Value| (µg/m³)", fontsize=12)
        ax.set_ylabel("Environmental Attribution Group", fontsize=12)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "global_group_importance.png", dpi=300)
        plt.close(fig)

        # 4. Group Importance by Season
        try:
            fig, ax = plt.subplots(figsize=(11, 6))
            grp_cols = [c for c in seasonal_df.columns if c.endswith("_shap")]
            season_melted = pd.melt(seasonal_df, id_vars=["season"], value_vars=grp_cols, var_name="group", value_name="mean_shap")
            season_melted["group"] = season_melted["group"].str.replace("_shap", "")
            sns.barplot(data=season_melted, x="season", y="mean_shap", hue="group", ax=ax, palette="Set2", edgecolor="black")
            ax.set_title("AtmosIQ Phase 4B: Group SHAP Contributions Across Seasons", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Season", fontsize=12)
            ax.set_ylabel("Mean Signed SHAP Contribution (µg/m³)", fontsize=12)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.legend(title="Attribution Group", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            fig.savefig(self.plot_dir / "group_importance_by_season.png", dpi=300)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not generate group_importance_by_season plot: {e}")

        # 5. High vs Normal Pollution Days
        fig, ax = plt.subplots(figsize=(10, 6))
        high_melted = pd.melt(high_analysis_df, id_vars=["attribution_group"], value_vars=["high_pollution_mean_shap", "normal_days_mean_shap"], var_name="day_type", value_name="mean_shap")
        high_melted["day_type"] = high_melted["day_type"].str.replace("_mean_shap", "").str.replace("_", " ").str.title()
        sns.barplot(data=high_melted, x="attribution_group", y="mean_shap", hue="day_type", ax=ax, palette="Dark2", edgecolor="black")
        ax.set_title("AtmosIQ Phase 4B: Group SHAP Comparison (Normal vs Top 10% High-Pollution Days)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Attribution Group", fontsize=12)
        ax.set_ylabel("Mean Signed SHAP Contribution (µg/m³)", fontsize=12)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xticks(rotation=15)
        plt.legend(title="Observation Group", loc="upper right")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "high_vs_normal_pollution_shap.png", dpi=300)
        plt.close(fig)

        # 6. Local Waterfall Plots for 5 Representative Dates
        for case_key, case_info in rep_dates_info.items():
            dt_str = case_info["date"]
            idx = case_info["row_index"]
            act = case_info["actual_pm25"]
            pred = case_info["predicted_pm25"]

            fig, ax = plt.subplots(figsize=(10, 6))
            top_feats = case_info["feature_contributions"].head(10).sort_values("abs_shap_value", ascending=True)
            colors_w = ["#2ecc71" if s > 0 else "#e74c3c" for s in top_feats["shap_value"]]
            ax.barh(top_feats["feature_name"], top_feats["shap_value"], color=colors_w, edgecolor="black", alpha=0.85)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"TreeSHAP Explanation: {case_key.replace('_', ' ').title()} ({dt_str})\nActual: {act:.1f} µg/m³ | Predicted: {pred:.1f} µg/m³ | Base: {base_value:.1f} µg/m³", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("SHAP Contribution (µg/m³)", fontsize=11)
            ax.set_ylabel("Top 10 Influential Features", fontsize=11)
            plt.tight_layout()
            fig.savefig(self.plot_dir / f"waterfall_{case_key}.png", dpi=300)
            plt.close(fig)

        logger.info(f"All Phase 4B diagnostic plots saved under {self.plot_dir}.")


if __name__ == "__main__":
    viz = VisualizationEnginePhase4B()
