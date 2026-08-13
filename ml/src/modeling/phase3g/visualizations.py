import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationsPhase3G")


class VisualizationEnginePhase3G:
    """
    AtmosIQ Phase 3G Visualization Engine.
    Generates all 7 required diagnostic plots under ml/experiments/phase3g/plots/.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase3g"):
        self.exp_dir = Path(exp_dir)
        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.optuna_dir = self.exp_dir / "optuna"
        self.metrics_dir = self.exp_dir / "metrics"
        self.preds_dir = self.exp_dir / "predictions"

    def generate_all_plots(self, best_cand: dict):
        """Generates all 7 required diagnostic plots."""
        logger.info("Generating Phase 3G diagnostic plots...")

        # 1. Optuna History
        trials_file = self.optuna_dir / "trials.csv"
        if trials_file.exists():
            trials_df = pd.read_csv(trials_file).dropna(subset=["val_mae"])

            plt.figure(figsize=(10, 5))
            plt.plot(trials_df["trial_id"], trials_df["val_mae"], "o-", color="teal", alpha=0.7, markersize=4)
            plt.title("Phase 3G: Optuna Optimization History Across Trials", fontweight="bold")
            plt.xlabel("Trial ID")
            plt.ylabel("Development Mean MAE (µg/m³)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "optuna_history.png", dpi=300)
            plt.close()

            # 2. Validation MAE by Trial
            plt.figure(figsize=(10, 5))
            plt.scatter(trials_df["trial_id"], trials_df["val_mae"], c=trials_df["val_mae"], cmap="viridis_r", alpha=0.8, edgecolors="k", s=35)
            plt.colorbar(label="Dev Mean MAE (µg/m³)")
            plt.title("Phase 3G: Development Validation MAE by Trial ID", fontweight="bold")
            plt.xlabel("Trial ID")
            plt.ylabel("Development Mean MAE (µg/m³)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "validation_mae_by_trial.png", dpi=300)
            plt.close()

        # 3. Model Comparison (Dev Mean MAE)
        comp_file = self.metrics_dir / "model_comparison.csv"
        if comp_file.exists():
            comp_df = pd.read_csv(comp_file).sort_values("Dev_Mean_MAE")

            plt.figure(figsize=(10, 5))
            labels = [f"{r['Model']}\n({r['Feature_Set']})" for _, r in comp_df.iterrows()]
            plt.barh(labels, comp_df["Dev_Mean_MAE"], color="darkblue", edgecolor="black")
            plt.axvline(32.7667, color="red", linestyle="--", label="Persistence Baseline (32.77 µg/m³)")
            plt.title("Phase 3G: Tuned Models Development Mean MAE Comparison", fontweight="bold")
            plt.xlabel("Development Mean MAE (µg/m³)")
            plt.gca().invert_yaxis()
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "model_comparison.png", dpi=300)
            plt.close()

        # 4. Fold Stability
        fold_file = self.metrics_dir / "fold_metrics.csv"
        if fold_file.exists():
            fold_df = pd.read_csv(fold_file)

            plt.figure(figsize=(11, 5))
            for m_name in fold_df["Model"].unique():
                sub = fold_df[fold_df["Model"] == m_name].sort_values("Val_Year")
                plt.plot(sub["Val_Year"], sub["MAE"], "o-", label=m_name, linewidth=2, markersize=7)

            plt.title("Phase 3G: Model Performance Stability Across Walk-Forward Folds", fontweight="bold")
            plt.xlabel("Validation Year")
            plt.ylabel("MAE (µg/m³)")
            plt.xticks([2022, 2023], ["2022 (Fold 1)", "2023 (Fold 2)"])
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "fold_stability.png", dpi=300)
            plt.close()

        # 5. Final Test Predictions Plots (Locked 2024 Test Set)
        final_pred_file = self.preds_dir / "final_test_predictions.csv"
        if final_pred_file.exists():
            final_df = pd.read_csv(final_pred_file)
            final_df["date"] = pd.to_datetime(final_df["date"])

            # A. Actual vs Pred Final
            plt.figure(figsize=(9, 5))
            plt.scatter(final_df["actual_pm25"], final_df["predicted_pm25"], color="darkgreen", alpha=0.6, edgecolors="k", s=35)
            max_v = max(final_df["actual_pm25"].max(), final_df["predicted_pm25"].max())
            min_v = min(final_df["actual_pm25"].min(), final_df["predicted_pm25"].min())
            plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Ideal Perfect Prediction (y=x)")
            plt.title(f"Final Production Model ({best_cand['Model']} on {best_cand['Feature_Set']}): Actual vs Predicted (2024 Test)", fontweight="bold")
            plt.xlabel("Actual PM2.5 (µg/m³)")
            plt.ylabel("Predicted PM2.5 (µg/m³)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / "actual_vs_pred_final.png", dpi=300)
            plt.close()

            # B. Residuals Over Time Final
            plt.figure(figsize=(11, 5))
            plt.plot(final_df["date"], final_df["residual"], color="teal", alpha=0.8)
            plt.axhline(0, color="black", linestyle="--")
            plt.title(f"Final Production Model ({best_cand['Model']}): Residuals Over Time (2024 Test Set)", fontweight="bold")
            plt.xlabel("Date")
            plt.ylabel("Residual (Actual - Predicted)")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "residuals_over_time_final.png", dpi=300)
            plt.close()

            # C. Residual Distribution Final
            plt.figure(figsize=(9, 5))
            plt.hist(final_df["residual"], bins=30, color="purple", edgecolor="black", alpha=0.7)
            plt.axvline(0, color="red", linestyle="--")
            plt.title(f"Final Production Model ({best_cand['Model']}): Residual Distribution (2024 Test Set)", fontweight="bold")
            plt.xlabel("Residual (µg/m³)")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "residual_distribution_final.png", dpi=300)
            plt.close()

        logger.info(f"All 7 diagnostic plots saved to: {self.plots_dir}")


if __name__ == "__main__":
    viz = VisualizationEnginePhase3G()
    viz.generate_all_plots({"Model": "XGBoost", "Feature_Set": "group_f_pm25_met_fire_transport"})
