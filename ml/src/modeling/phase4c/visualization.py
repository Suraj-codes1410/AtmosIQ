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

logger = setup_logger("VisualizationPhase4C")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"


class VisualizationEnginePhase4C:
    """
    AtmosIQ Phase 4C Visualization Engine.
    Generates 10 high-resolution diagnostic plots under ml/experiments/phase4c/plots/.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)
        self.plot_dir = self.exp_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self, df: pd.DataFrame, group_shap_df: pd.DataFrame, seasonal_df: pd.DataFrame, temp_df: pd.DataFrame, catalog_df: pd.DataFrame, conflict_df: pd.DataFrame, conf_df: pd.DataFrame):
        """Generates all 10 Phase 4C diagnostic plots."""
        logger.info("Generating Phase 4C Environmental Validation Plots...")

        fire_col = "fire_hotspot_count_lag_1d" if "fire_hotspot_count_lag_1d" in df.columns else "fire_hotspot_count_roll_mean_7d"
        wind_col = "wind_speed_kmh_lag_1d" if "wind_speed_kmh_lag_1d" in df.columns else "wind_speed_kmh_roll_mean_7d"

        # 1. Biomass SHAP vs Fire Activity
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.regplot(data=pd.DataFrame({"fire": df[fire_col], "shap": group_shap_df["biomass_burning_shap"]}), x="fire", y="shap", ax=ax, color="#e74c3c", scatter_kws={"alpha": 0.3}, line_kws={"color": "black", "linewidth": 2})
        ax.set_title("Biomass Burning SHAP vs Upwind Satellite Fire Hotspots", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Upwind Satellite Fire Hotspots Count", fontsize=11)
        ax.set_ylabel("Biomass Burning SHAP Contribution (µg/m³)", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "biomass_shap_vs_fire_activity.png", dpi=300)
        plt.close(fig)

        # 2. Biomass SHAP Distribution by Fire Quantile
        fig, ax = plt.subplots(figsize=(8, 5))
        q25, q75 = df[fire_col].quantile(0.25), df[fire_col].quantile(0.75)
        fire_cats = pd.cut(df[fire_col], bins=[-np.inf, q25, q75, np.inf], labels=["Low Fire (<=25%)", "Normal Fire (25-75%)", "High Fire (>=75%)"])
        plot_df = pd.DataFrame({"category": fire_cats, "shap": group_shap_df["biomass_burning_shap"]})
        sns.boxplot(data=plot_df, x="category", y="shap", hue="category", ax=ax, palette="YlOrRd", legend=False)
        ax.set_title("Biomass Burning SHAP Distribution Across Fire Activity Quantiles", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Fire Activity Quantile", fontsize=11)
        ax.set_ylabel("Biomass Burning SHAP (µg/m³)", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "biomass_shap_by_fire_quantile.png", dpi=300)
        plt.close(fig)

        # 3. Wind SHAP vs Wind Speed
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.regplot(data=pd.DataFrame({"wind": df[wind_col], "shap": group_shap_df["wind_ventilation_shap"]}), x="wind", y="shap", ax=ax, color="#3498db", scatter_kws={"alpha": 0.3}, line_kws={"color": "black", "linewidth": 2})
        ax.set_title("Wind / Ventilation SHAP vs Surface Wind Speed", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Surface Wind Speed (km/h)", fontsize=11)
        ax.set_ylabel("Wind / Ventilation SHAP Contribution (µg/m³)", fontsize=11)
        ax.axhline(0, color="gray", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "wind_shap_vs_wind_speed.png", dpi=300)
        plt.close(fig)

        # 4. Wind SHAP vs Ventilation Regimes
        fig, ax = plt.subplots(figsize=(8, 5))
        wind_cats = pd.cut(df[wind_col], bins=[-np.inf, 5.0, 12.0, np.inf], labels=["Stagnation (<=5km/h)", "Moderate (5-12km/h)", "Dispersion (>=12km/h)"])
        plot_w_df = pd.DataFrame({"regime": wind_cats, "shap": group_shap_df["wind_ventilation_shap"]})
        sns.barplot(data=plot_w_df, x="regime", y="shap", hue="regime", ax=ax, palette="Blues_r", edgecolor="black", legend=False)
        ax.set_title("Wind SHAP Contribution Across Atmospheric Ventilation Regimes", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Ventilation Regime", fontsize=11)
        ax.set_ylabel("Mean Wind SHAP (µg/m³)", fontsize=11)
        ax.axhline(0, color="black", linestyle="--")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "wind_shap_vs_ventilation.png", dpi=300)
        plt.close(fig)

        # 5. Seasonal Group Attribution Heatmap
        fig, ax = plt.subplots(figsize=(9, 6))
        pivot_seasonal = seasonal_df.pivot(index="season", columns="attribution_group", values="mean_signed_shap")
        sns.heatmap(pivot_seasonal, annot=True, fmt="+.2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Mean Signed SHAP (µg/m³)"})
        ax.set_title("AtmosIQ Phase 4C: Seasonal Attribution Heatmap", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Attribution Group", fontsize=11)
        ax.set_ylabel("Season", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "seasonal_group_attribution_heatmap.png", dpi=300)
        plt.close(fig)

        # 6. Yearly Attribution Stability (2020-2024)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=temp_df, x="year", y="mean_abs_shap", hue="attribution_group", ax=ax, palette="Set2", edgecolor="black")
        ax.set_title("Multi-Year Group Attribution Stability (2020-2024)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Mean |SHAP Value| (µg/m³)", fontsize=11)
        plt.legend(title="Attribution Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "yearly_attribution_stability.png", dpi=300)
        plt.close(fig)

        # 7. Pollution Event Attribution Timeline (Top 5 Events)
        fig, ax = plt.subplots(figsize=(10, 5))
        top5_cat = catalog_df.head(5)
        x_indices = np.arange(len(top5_cat))
        width = 0.18
        groups_list = ["pm25_persistence_shap", "biomass_burning_shap", "wind_ventilation_shap", "meteorology_shap"]
        colors_list = ["#34495e", "#e74c3c", "#3498db", "#2ecc71"]

        for i, grp_col in enumerate(groups_list):
            ax.bar(x_indices + (i - 1.5) * width, top5_cat[grp_col], width, label=grp_col.replace("_shap", "").replace("_", " ").title(), color=colors_list[i], edgecolor="black")

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{r['event_id']}\n({r['event_start']})" for _, r in top5_cat.iterrows()], rotation=0)
        ax.set_title("Attribution Breakdown Across Top Pollution Episodes", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Pollution Event ID & Start Date", fontsize=11)
        ax.set_ylabel("Group SHAP Contribution (µg/m³)", fontsize=11)
        plt.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "pollution_event_attribution_timeline.png", dpi=300)
        plt.close(fig)

        # 8. Fire Activity + Biomass SHAP Timeline (Post-Monsoon 2024 Episode)
        fig, ax1 = plt.subplots(figsize=(11, 5))
        post_m_2024 = df[df["date"].str.startswith("2024-10") | df["date"].str.startswith("2024-11")].copy()
        post_m_shap = group_shap_df.loc[post_m_2024.index]

        color1 = "#e74c3c"
        ax1.set_xlabel("Date (Post-Monsoon 2024)", fontsize=11)
        ax1.set_ylabel("Biomass Burning SHAP (µg/m³)", color=color1, fontsize=11)
        ax1.plot(post_m_2024["date"], post_m_shap["biomass_burning_shap"], color=color1, linewidth=2, label="Biomass SHAP")
        ax1.tick_params(axis="y", labelcolor=color1)
        plt.xticks(rotation=45)

        ax2 = ax1.twinx()
        color2 = "#d35400"
        ax2.set_ylabel("Upwind Satellite Fire Hotspots Count", color=color2, fontsize=11)
        ax2.bar(post_m_2024["date"], post_m_2024[fire_col], color=color2, alpha=0.3, label="Satellite Fire Count")
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("Timeline Comparison: Satellite Fire Counts vs Model Biomass SHAP (Oct-Nov 2024)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "fire_activity_and_biomass_shap_timeline.png", dpi=300)
        plt.close(fig)

        # 9. Attribution Conflict Cases
        fig, ax = plt.subplots(figsize=(9, 5))
        if len(conflict_df) > 0:
            c_counts = conflict_df["conflict_type"].value_counts()
            ax.barh(c_counts.index, c_counts.values, color="#e67e22", edgecolor="black")
        else:
            ax.text(0.5, 0.5, "No Counter-Evidence Conflicts Detected", ha="center", va="center", fontsize=14)
        ax.set_title("Identified Attribution Counter-Evidence Conflict Cases", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Observation Count", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.plot_dir / "attribution_conflict_cases.png", dpi=300)
        plt.close(fig)

        # 10. Attribution Confidence Distribution
        fig, ax = plt.subplots(figsize=(7, 5))
        conf_counts = conf_df["confidence_level"].value_counts()[["High", "Moderate", "Low"]]
        colors_conf = ["#2ecc71", "#f39c12", "#e74c3c"]
        ax.bar(conf_counts.index, conf_counts.values, color=colors_conf, edgecolor="black", alpha=0.85)
        ax.set_title("Environmental Support Confidence Level Distribution", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Confidence Level", fontsize=11)
        ax.set_ylabel("Observation Count", fontsize=11)
        for i, v in enumerate(conf_counts.values):
            ax.text(i, v + 20, f"{v} ({v/len(conf_df)*100:.1f}%)", ha="center", fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.plot_dir / "attribution_confidence_distribution.png", dpi=300)
        plt.close(fig)

        logger.info(f"All 10 Phase 4C diagnostic plots saved under {self.plot_dir}.")


if __name__ == "__main__":
    viz = VisualizationEnginePhase4C()
