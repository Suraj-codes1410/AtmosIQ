import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationPhase4H")


class VisualizationEnginePhase4H:
    """
    Publication-Quality Visualization Generator for Phase 4H.
    Generates 10 standard evaluation figures in ml/experiments/phase4h/plots/.
    """

    def __init__(self, plot_dir: Path):
        self.plot_dir = plot_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    def generate_all_plots(
        self,
        df_summary: pd.DataFrame,
        df_folds: pd.DataFrame,
        model_predictions: dict,
        stat_results: dict,
        ablation_df: pd.DataFrame,
        seasonal_df: pd.DataFrame,
        extreme_df: pd.DataFrame,
        feature_importance_df: pd.DataFrame
    ):
        logger.info("Generating Phase 4H Publication Plots...")

        # Plot 1: Model MAE Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_summary, x='model_name', y='mean_mae', hue='feature_set', palette='Blues_r')
        plt.title("Master Model Evaluation — Mean Out-of-Sample MAE Across Folds", fontsize=13, fontweight='bold')
        plt.ylabel("Mean Test MAE (µg/m³)")
        plt.xlabel("Candidate Model")
        plt.xticks(rotation=15)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(self.plot_dir / "1_model_mae_comparison.png", dpi=300)
        plt.close()

        # Plot 2: Model R2 Comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_summary, x='model_name', y='mean_r2', hue='feature_set', palette='Greens_r')
        plt.title("Master Model Evaluation — Mean Out-of-Sample R² Across Folds", fontsize=13, fontweight='bold')
        plt.ylabel("Mean Test R²")
        plt.xlabel("Candidate Model")
        plt.xticks(rotation=15)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(self.plot_dir / "2_model_r2_comparison.png", dpi=300)
        plt.close()

        # Plot 3: Fold-by-Fold Performance
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_folds, x='test_year', y='test_mae', hue='model_name', style='feature_set', markers=True, dashes=False, linewidth=2.5)
        plt.title("Temporal Walk-Forward Stability — Fold-by-Fold Test MAE (2022–2024)", fontsize=13, fontweight='bold')
        plt.ylabel("Test MAE (µg/m³)")
        plt.xlabel("Walk-Forward Test Year")
        plt.xticks([2022, 2023, 2024])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plot_dir / "3_fold_by_fold_performance.png", dpi=300)
        plt.close()

        # Plot 4: Prediction Error Distributions
        plt.figure(figsize=(10, 6))
        for key, df_pred in model_predictions.items():
            if "Frozen_RF" in key or "RandomForest__Candidate_B" in key or "XGBoost__Candidate_B" in key:
                errors = df_pred['y_pred'] - df_pred['y_true']
                sns.kdeplot(errors, label=key, linewidth=2)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
        plt.title("Out-of-Sample Prediction Error Density Distributions (y_pred - y_true)", fontsize=13, fontweight='bold')
        plt.xlabel("Prediction Error (µg/m³)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_dir / "4_prediction_error_distributions.png", dpi=300)
        plt.close()

        # Plot 5: v2 vs v3 Error Difference
        plt.figure(figsize=(12, 5))
        ctrl_key = [k for k in model_predictions.keys() if "Frozen_RF" in k][0]
        v3_keys = [k for k in model_predictions.keys() if "Candidate_B" in k]
        best_v3_key = v3_keys[0] if v3_keys else list(model_predictions.keys())[0]

        df_ctrl = model_predictions[ctrl_key].sort_values("date")
        df_v3 = model_predictions[best_v3_key].sort_values("date")
        merged = pd.merge(df_ctrl, df_v3, on=["date", "y_true"], suffixes=("_ctrl", "_v3"))
        merged['date'] = pd.to_datetime(merged['date'])

        diff = np.abs(merged['y_pred_v3'] - merged['y_true']) - np.abs(merged['y_pred_ctrl'] - merged['y_true'])
        plt.plot(merged['date'], diff, color='purple', alpha=0.6, linewidth=1)
        plt.axhline(0, color='black', linestyle='--', alpha=0.8)
        plt.title(f"Daily Error Difference (|e_v3| - |e_v2|) Across Test Years (Negative = v3 Improved)", fontsize=12, fontweight='bold')
        plt.ylabel("Error Difference (µg/m³)")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(self.plot_dir / "5_v2_vs_v3_error_difference.png", dpi=300)
        plt.close()

        # Plot 6: Seasonal Performance
        if len(seasonal_df) > 0:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=seasonal_df, x='season', y='mae', hue='model_name', palette='Set2')
            plt.title("Seasonal Performance Breakdown (Winter, Summer, Monsoon, Post-Monsoon)", fontsize=13, fontweight='bold')
            plt.ylabel("Season Test MAE (µg/m³)")
            plt.xlabel("Season")
            plt.tight_layout()
            plt.savefig(self.plot_dir / "6_seasonal_performance.png", dpi=300)
            plt.close()

        # Plot 7: Extreme Event Performance
        if len(extreme_df) > 0:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=extreme_df, x='regime', y='mae', hue='model_name', palette='Reds_r')
            plt.title("High Pollution Event Performance Breakdown", fontsize=13, fontweight='bold')
            plt.ylabel("Test MAE (µg/m³)")
            plt.xlabel("Pollution Regime")
            plt.xticks(rotation=10)
            plt.tight_layout()
            plt.savefig(self.plot_dir / "7_extreme_event_performance.png", dpi=300)
            plt.close()

        # Plot 8: External Feature Ablation
        if len(ablation_df) > 0:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=ablation_df, x='ablation_config', y='mean_mae', palette='Purples_r')
            plt.title("External Environmental Feature Group Ablation Study", fontsize=13, fontweight='bold')
            plt.ylabel("Mean Test MAE (µg/m³)")
            plt.xlabel("Ablation Configuration")
            plt.xticks(rotation=15)
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
            plt.tight_layout()
            plt.savefig(self.plot_dir / "8_external_feature_ablation.png", dpi=300)
            plt.close()

        # Plot 9: Feature Importance
        if len(feature_importance_df) > 0:
            plt.figure(figsize=(10, 8))
            top20 = feature_importance_df.head(20)
            sns.barplot(data=top20, x='importance', y='feature', palette='viridis')
            plt.title("Top 20 Feature Importances (Best Candidate Model)", fontsize=13, fontweight='bold')
            plt.xlabel("Gini Feature Importance")
            plt.ylabel("Feature Name")
            plt.tight_layout()
            plt.savefig(self.plot_dir / "9_feature_importance.png", dpi=300)
            plt.close()

        # Plot 10: Bootstrap Confidence Intervals
        if stat_results:
            plt.figure(figsize=(8, 4))
            ci_low = stat_results["delta_mae_ci_lower"]
            ci_high = stat_results["delta_mae_ci_upper"]
            mean_diff = stat_results["delta_mae_mean"]

            plt.errorbar(mean_diff, 0, xerr=[[mean_diff - ci_low], [ci_high - mean_diff]], fmt='o', color='crimson', ecolor='darkred', elinewidth=3, capsize=8, markersize=8)
            plt.axvline(0, color='black', linestyle='--', alpha=0.7)
            plt.yticks([])
            plt.xlabel("95% Bootstrap Confidence Interval for ΔMAE (µg/m³)")
            plt.title(f"Bootstrap ΔMAE CI: [{ci_low:.4f}, {ci_high:.4f}] (Mean={mean_diff:.4f})", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plot_dir / "10_bootstrap_confidence_intervals.png", dpi=300)
            plt.close()

        logger.info(f"All 10 publication-quality plots generated in {self.plot_dir}.")
