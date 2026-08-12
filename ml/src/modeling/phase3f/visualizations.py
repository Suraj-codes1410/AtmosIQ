import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationsPhase3F")


class VisualizationEnginePhase3F:
    """
    AtmosIQ Phase 3F Visualization Engine.
    Generates all 9 diagnostic plots under ml/experiments/phase3f/plots/.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase3f"):
        self.exp_dir = Path(exp_dir)
        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.exp_dir / "feature_group_metrics.csv"
        self.overfit_file = self.exp_dir / "overfitting_analysis.csv"
        self.stability_file = self.exp_dir / "cross_fold_summary.csv"
        self.inc_file = self.exp_dir / "incremental_information.csv"

    def generate_all_plots(self):
        """Generates all 9 required plots."""
        logger.info("Generating Phase 3F diagnostic plots...")

        assert self.metrics_file.exists(), f"Metrics file missing: {self.metrics_file}"
        metrics_df = pd.read_csv(self.metrics_file)
        overfit_df = pd.read_csv(self.overfit_file)
        stability_df = pd.read_csv(self.stability_file)
        inc_df = pd.read_csv(self.inc_file)

        xgb_stab = stability_df[stability_df["Model"] == "XGBoost"].sort_values("Dev_Mean_MAE")

        # 1. Feature Group vs Validation MAE
        plt.figure(figsize=(10, 5))
        plt.barh(xgb_stab["Feature_Group"], xgb_stab["Dev_Mean_MAE"], color="teal", edgecolor="black")
        plt.axvline(32.7667, color="red", linestyle="--", label="Persistence Baseline (32.77 µg/m³)")
        plt.title("Phase 3F: Feature Group vs Development Mean MAE (XGBoost)", fontweight="bold")
        plt.xlabel("Mean MAE (µg/m³)")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "feature_group_vs_val_mae.png", dpi=300)
        plt.close()

        # 2. Feature Group vs Validation R2
        plt.figure(figsize=(10, 5))
        xgb_r2 = xgb_stab.sort_values("Dev_Mean_R2", ascending=False)
        plt.barh(xgb_r2["Feature_Group"], xgb_r2["Dev_Mean_R2"], color="darkblue", edgecolor="black")
        plt.axvline(0.7280, color="red", linestyle="--", label="Persistence Baseline (0.7280)")
        plt.title("Phase 3F: Feature Group vs Development Mean R² (XGBoost)", fontweight="bold")
        plt.xlabel("Mean R²")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "feature_group_vs_val_r2.png", dpi=300)
        plt.close()

        # 3. Feature Count vs MAE
        plt.figure(figsize=(9, 5))
        xgb_sorted_cnt = xgb_stab.sort_values("Feature_Count")
        plt.plot(xgb_sorted_cnt["Feature_Count"], xgb_sorted_cnt["Dev_Mean_MAE"], "o-", color="darkgreen", linewidth=2, markersize=7)
        plt.title("Phase 3F: Feature Count vs Development Mean MAE (XGBoost)", fontweight="bold")
        plt.xlabel("Feature Count")
        plt.ylabel("Mean MAE (µg/m³)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "feature_count_vs_mae.png", dpi=300)
        plt.close()

        # 4. Feature Count vs R2
        plt.figure(figsize=(9, 5))
        plt.plot(xgb_sorted_cnt["Feature_Count"], xgb_sorted_cnt["Dev_Mean_R2"], "s-", color="purple", linewidth=2, markersize=7)
        plt.title("Phase 3F: Feature Count vs Development Mean R² (XGBoost)", fontweight="bold")
        plt.xlabel("Feature Count")
        plt.ylabel("Mean R²")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "feature_count_vs_r2.png", dpi=300)
        plt.close()

        # 5. Model x Feature Group Heatmap (MAE) using Matplotlib
        plt.figure(figsize=(11, 6))
        pivot_mae = stability_df.pivot(index="Feature_Group", columns="Model", values="Dev_Mean_MAE")
        im = plt.imshow(pivot_mae.values, cmap="YlGnBu_r")
        plt.colorbar(im, label="Development Mean MAE (µg/m³)")
        plt.xticks(np.arange(len(pivot_mae.columns)), pivot_mae.columns, rotation=15)
        plt.yticks(np.arange(len(pivot_mae.index)), pivot_mae.index)

        # Annotate heatmap values
        for i in range(len(pivot_mae.index)):
            for j in range(len(pivot_mae.columns)):
                val = pivot_mae.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)

        plt.title("Phase 3F: Model × Feature Group Development Mean MAE Heatmap", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "model_x_feature_group_heatmap.png", dpi=300)
        plt.close()

        # 6. Incremental MAE Improvement
        xgb_inc = inc_df[inc_df["Model"] == "XGBoost"]
        plt.figure(figsize=(10, 5))
        labels_inc = [f"{r['Baseline_Group']}\n→ {r['New_Group']}" for _, r in xgb_inc.iterrows()]
        colors = ["green" if v > 0 else "red" for v in xgb_inc["Delta_MAE"]]
        plt.barh(labels_inc, xgb_inc["Delta_MAE"], color=colors, edgecolor="black")
        plt.axvline(0, color="black", linestyle="--")
        plt.title("Phase 3F: Incremental MAE Improvement (Δ MAE µg/m³) - XGBoost", fontweight="bold")
        plt.xlabel("Δ MAE (Positive is Better)")
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "incremental_mae_improvement.png", dpi=300)
        plt.close()

        # 7. Incremental R2 Improvement
        plt.figure(figsize=(10, 5))
        colors_r2 = ["green" if v > 0 else "red" for v in xgb_inc["Delta_R2"]]
        plt.barh(labels_inc, xgb_inc["Delta_R2"], color=colors_r2, edgecolor="black")
        plt.axvline(0, color="black", linestyle="--")
        plt.title("Phase 3F: Incremental R² Improvement (Δ R²) - XGBoost", fontweight="bold")
        plt.xlabel("Δ R² (Positive is Better)")
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "incremental_r2_improvement.png", dpi=300)
        plt.close()

        # 8. Generalization Gap (Train R2 vs Eval R2)
        plt.figure(figsize=(10, 5))
        dev_of = overfit_df[(overfit_df["Is_Holdout"] == False) & (overfit_df["Model"] == "XGBoost")]
        grp_of = dev_of.groupby("Feature_Group")[["Train_R2", "Eval_R2"]].mean().reset_index()
        bar_width = 0.35
        x = np.arange(len(grp_of))
        plt.bar(x - bar_width/2, grp_of["Train_R2"], bar_width, label="Train R²", color="navy")
        plt.bar(x + bar_width/2, grp_of["Eval_R2"], bar_width, label="Development Eval R²", color="orange")
        plt.xticks(x, grp_of["Feature_Group"], rotation=30, ha="right")
        plt.title("Phase 3F: Train R² vs Development Eval R² (XGBoost Generalization Gap)", fontweight="bold")
        plt.ylabel("R² Score")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "generalization_gap.png", dpi=300)
        plt.close()

        # 9. Fold Stability Across 2022, 2023, and 2024
        plt.figure(figsize=(11, 5))
        top_groups = ["group_b_pm25_history", "group_c_pm25_meteorology", "group_f_pm25_met_fire_transport", "group_g_full_safe"]
        sub_metrics = metrics_df[(metrics_df["Model"] == "XGBoost") & (metrics_df["Feature_Group"].isin(top_groups))]

        for g in top_groups:
            g_sub = sub_metrics[sub_metrics["Feature_Group"] == g].sort_values("Eval_Year")
            plt.plot(g_sub["Eval_Year"], g_sub["MAE"], "o-", label=g, linewidth=2, markersize=7)

        plt.title("Phase 3F: Fold Stability Across 2022, 2023, and 2024 Holdout (XGBoost)", fontweight="bold")
        plt.xlabel("Evaluation Year")
        plt.ylabel("MAE (µg/m³)")
        plt.xticks([2022, 2023, 2024], ["2022 (Fold 1)", "2023 (Fold 2)", "2024 (Fold 3 Holdout)"])
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "fold_stability.png", dpi=300)
        plt.close()

        logger.info(f"All 9 diagnostic plots saved to: {self.plots_dir}")


if __name__ == "__main__":
    viz = VisualizationEnginePhase3F()
    viz.generate_all_plots()
