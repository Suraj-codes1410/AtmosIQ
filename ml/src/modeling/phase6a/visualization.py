import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationPhase6A")


class VisualizationEnginePhase6A:
    """
    Publication-Quality Visualization Generator for Phase 6A.
    Generates 12 clean, professional diagnostic plots in ml/experiments/phase6a/plots/.
    """

    def __init__(self, df_preds: pd.DataFrame, df_intervals: pd.DataFrame, df_metrics: pd.DataFrame, df_cond: pd.DataFrame):
        self.df_preds = df_preds.copy()
        self.df_preds['date_dt'] = pd.to_datetime(self.df_preds['date'])
        self.df_intervals = df_intervals.copy()
        self.df_intervals['date_dt'] = pd.to_datetime(self.df_intervals['date'])
        self.df_metrics = df_metrics.copy()
        self.df_cond = df_cond.copy()

        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 14
        })

    def generate_all_plots(self, plot_dir: Path):
        logger.info(f"Generating 12 publication-quality plots in {plot_dir}...")
        plot_dir.mkdir(parents=True, exist_ok=True)

        residuals = self.df_preds['residual'].values

        # 1. Residual Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, kde=True, stat="density", color="steelblue", bins=40, ax=ax, label="Empirical Residuals")
        # Gaussian fit curve
        mu, std = np.mean(residuals), np.std(residuals)
        x_grid = np.linspace(min(residuals), max(residuals), 200)
        ax.plot(x_grid, stats.norm.pdf(x_grid, mu, std), color="crimson", linestyle="--", linewidth=2, label=f"Normal Fit (μ={mu:.1f}, σ={std:.1f})")
        ax.set_title("Empirical Residual Distribution (Out-of-Sample 2022–2024)")
        ax.set_xlabel("Residual e = y_obs - y_pred (µg/m³)")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "residual_distribution.png", dpi=150)
        plt.close(fig)

        # 2. Residual Q-Q Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot of Out-of-Sample Residuals")
        ax.get_lines()[0].set_color("steelblue")
        ax.get_lines()[1].set_color("crimson")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Ordered Residuals (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "residual_quantiles.png", dpi=150)
        plt.close(fig)

        # 3. Residuals Over Time
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(self.df_preds['date_dt'], self.df_preds['residual'], color="teal", alpha=0.7, linewidth=1, label="Daily Residual")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title("Walk-Forward Prediction Residuals Over Time (2022–2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Residual (µg/m³)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "residuals_over_time.png", dpi=150)
        plt.close(fig)

        # 4. Residuals vs Prediction (Heteroscedasticity Test)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=self.df_preds, x='predicted_pm25', y='residual', hue='season', alpha=0.6, palette='viridis', ax=ax)
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
        ax.set_title("Residuals vs. Predicted PM2.5 (Heteroscedasticity Analysis)")
        ax.set_xlabel("Predicted PM2.5 (µg/m³)")
        ax.set_ylabel("Residual (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "residuals_vs_prediction.png", dpi=150)
        plt.close(fig)

        # 5. Residual Distribution by Season
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.df_preds, x='season', y='residual', palette='Set2', ax=ax)
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
        ax.set_title("Empirical Residual Spread Across Seasons")
        ax.set_xlabel("Season")
        ax.set_ylabel("Residual (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "residual_distribution_by_season.png", dpi=150)
        plt.close(fig)

        # 6. Residual Distribution by Year
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=self.df_preds, x='year', y='residual', palette='Pastel1', ax=ax)
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
        ax.set_title("Residual Stability Across Evaluation Years")
        ax.set_xlabel("Evaluation Year")
        ax.set_ylabel("Residual (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "residual_distribution_by_year.png", dpi=150)
        plt.close(fig)

        # 7. Residual Distribution by Pollution Regime
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.df_preds, x='pollution_regime', y='residual', order=["Low", "Moderate", "High", "Extreme"], palette='YlOrRd', ax=ax)
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
        ax.set_title("Residual Dispersion by Pollution Regime")
        ax.set_xlabel("Pollution Regime")
        ax.set_ylabel("Residual (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "residual_distribution_by_pollution_regime.png", dpi=150)
        plt.close(fig)

        # 8. Coverage Comparison across Methods
        # Average metrics across folds per method and nominal coverage
        df_cov_agg = self.df_metrics.groupby(['method', 'nominal_coverage'])['empirical_coverage'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=df_cov_agg, x='nominal_coverage', y='empirical_coverage', hue='method', palette='tab10', ax=ax)
        # Add diagonal/target reference lines
        for x_val in [0.80, 0.90, 0.95]:
            ax.axhline(x_val, color="gray", linestyle=":", alpha=0.5)
        ax.set_title("Empirical vs. Nominal Coverage Comparison across Baseline Methods")
        ax.set_xlabel("Nominal Coverage Level")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.5, 1.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "coverage_comparison.png", dpi=150)
        plt.close(fig)

        # 9. Interval Width Comparison
        df_w_agg = self.df_metrics.groupby(['method', 'nominal_coverage'])['mean_width_ugm3'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=df_w_agg, x='nominal_coverage', y='mean_width_ugm3', hue='method', palette='tab10', ax=ax)
        ax.set_title("Mean Prediction Interval Width (MPIW) Comparison")
        ax.set_xlabel("Nominal Coverage Level")
        ax.set_ylabel("MPIW (µg/m³)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "interval_width_comparison.png", dpi=150)
        plt.close(fig)

        # 10. Interval Coverage by Regime (90% Nominal)
        sub_reg = self.df_cond[(self.df_cond['dimension'] == 'Pollution_Regime') & (self.df_cond['nominal_coverage'] == 0.90)]
        if not sub_reg.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=sub_reg, x='slice_name', y='empirical_coverage', hue='method', order=["Low", "Moderate", "High", "Extreme"], palette='tab10', ax=ax)
            ax.axhline(0.90, color="crimson", linestyle="--", linewidth=1.5, label="Nominal 90% Target")
            ax.set_title("Empirical Coverage by Pollution Regime (Nominal 90%)")
            ax.set_xlabel("Pollution Regime")
            ax.set_ylabel("Empirical Coverage")
            ax.set_ylim(0.4, 1.05)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            fig.savefig(plot_dir / "interval_coverage_by_regime.png", dpi=150)
            plt.close(fig)

        # 11. Extreme Pollution Interval Performance
        sub_ext = self.df_cond[(self.df_cond['dimension'] == 'Extreme_Subset') & (self.df_cond['nominal_coverage'] == 0.90)]
        if not sub_ext.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=sub_ext, x='method', y='empirical_coverage', palette='Reds_r', ax=ax)
            ax.axhline(0.90, color="blue", linestyle="--", label="Nominal 90% Target")
            ax.set_title("Interval Coverage during Extreme Episodes (PM2.5 ≥ 150 µg/m³)")
            ax.set_xlabel("Method")
            ax.set_ylabel("Empirical Coverage")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
            ax.legend()
            plt.tight_layout()
            fig.savefig(plot_dir / "extreme_pollution_interval_performance.png", dpi=150)
            plt.close(fig)

        # 12. Representative Prediction Intervals (Post-Monsoon / Winter 2024 Window)
        sub_window = self.df_intervals[
            (self.df_intervals['method'] == 'empirical_residual_global') &
            (self.df_intervals['nominal_coverage'] == 0.90) &
            (self.df_intervals['date_dt'] >= '2024-10-15') &
            (self.df_intervals['date_dt'] <= '2024-12-15')
        ].sort_values('date_dt')

        if not sub_window.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(sub_window['date_dt'], sub_window['observed_pm25'], color="black", marker='o', markersize=3, linewidth=1.5, label="Observed PM2.5")
            ax.plot(sub_window['date_dt'], sub_window['predicted_pm25'], color="dodgerblue", linestyle="--", linewidth=1.5, label="Predicted PM2.5")
            ax.fill_between(sub_window['date_dt'], sub_window['lower_bound'], sub_window['upper_bound'], color="skyblue", alpha=0.4, label="90% Prediction Interval (Empirical)")
            ax.set_title("Representative 90% Prediction Intervals (Delhi Stubble & Winter Peak 2024)")
            ax.set_xlabel("Date")
            ax.set_ylabel("PM2.5 (µg/m³)")
            ax.legend(loc="upper left")
            plt.tight_layout()
            fig.savefig(plot_dir / "representative_prediction_intervals.png", dpi=150)
            plt.close(fig)

        logger.info("All 12 publication-quality plots generated cleanly.")
