import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationPhase6B")


class VisualizationEnginePhase6B:
    """
    Publication-Quality Visualization Generator for Phase 6B.
    Generates 14 diagnostic figures in ml/experiments/phase6b/plots/.
    """

    def __init__(
        self,
        df_boot: pd.DataFrame,
        df_seed: pd.DataFrame,
        df_intervals: pd.DataFrame,
        df_quintiles: pd.DataFrame,
        df_regime: pd.DataFrame,
        df_seasonal: pd.DataFrame,
        df_yearly: pd.DataFrame,
        df_extreme: pd.DataFrame,
        df_sens: pd.DataFrame,
        df_disc: pd.DataFrame
    ):
        self.df_boot = df_boot.copy()
        self.df_boot['date_dt'] = pd.to_datetime(self.df_boot['date'])
        self.df_seed = df_seed.copy()
        self.df_intervals = df_intervals.copy()
        self.df_quintiles = df_quintiles.copy()
        self.df_regime = df_regime.copy()
        self.df_seasonal = df_seasonal.copy()
        self.df_yearly = df_yearly.copy()
        self.df_extreme = df_extreme.copy()
        self.df_sens = df_sens.copy()
        self.df_disc = df_disc.copy()

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
        logger.info(f"Generating 14 publication-quality figures in {plot_dir}...")
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ensemble Prediction Spread Time Series
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(self.df_boot['date_dt'], self.df_boot['ensemble_std'], color="indigo", alpha=0.8, linewidth=1, label="Bootstrap Ensemble Spread (σ)")
        ax.set_title("Daily Ensemble Prediction Spread (Bootstrap B=30, 2022–2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Ensemble Spread (µg/m³)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "ensemble_prediction_spread.png", dpi=150)
        plt.close(fig)

        # 2. Ensemble Spread vs. Absolute Error Scatter
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=self.df_boot, x='ensemble_std', y='absolute_error', hue='pollution_regime', palette='tab10', alpha=0.6, ax=ax)
        # Trend line
        sns.regplot(data=self.df_boot, x='ensemble_std', y='absolute_error', scatter=False, color="black", ax=ax)
        ax.set_title("Ensemble Spread vs. Actual Absolute Prediction Error")
        ax.set_xlabel("Ensemble Spread σ (µg/m³)")
        ax.set_ylabel("Absolute Prediction Error (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "ensemble_spread_vs_absolute_error.png", dpi=150)
        plt.close(fig)

        # 3. Spread-Error Quantiles (MAE across quintiles)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df_quintiles, x='spread_quintile', y='mae_ugm3', hue='spread_quintile', palette='Blues_r', legend=False, ax=ax)
        ax.set_title("Prediction Error (MAE) Grouped by Ensemble Spread Quintile")
        ax.set_xlabel("Ensemble Spread Quintile")
        ax.set_ylabel("MAE (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "spread_error_quantiles.png", dpi=150)
        plt.close(fig)

        # 4. Calibration Curve
        fig, ax = plt.subplots(figsize=(6, 6))
        nom_levels = [0.80, 0.90, 0.95]
        sub_boot = self.df_intervals[self.df_intervals['method'] == 'bootstrap_clipped']
        emp_boot = [float(sub_boot[sub_boot['nominal_coverage'] == n]['covered'].mean()) for n in nom_levels]
        sub_seed = self.df_intervals[self.df_intervals['method'] == 'seed_clipped']
        emp_seed = [float(sub_seed[sub_seed['nominal_coverage'] == n]['covered'].mean()) for n in nom_levels]

        ax.plot(nom_levels, nom_levels, color="gray", linestyle="--", label="Ideal Calibration")
        ax.plot(nom_levels, emp_boot, marker='o', linewidth=2, color="teal", label="Bootstrap Ensemble (Clipped)")
        ax.plot(nom_levels, emp_seed, marker='s', linewidth=2, color="darkorange", label="Seed Ensemble (Clipped)")
        ax.set_title("Prediction Interval Calibration Curve")
        ax.set_xlabel("Nominal Coverage Level")
        ax.set_ylabel("Empirical Coverage Level")
        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(0.70, 1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "calibration_curve.png", dpi=150)
        plt.close(fig)

        # 5. Coverage by Regime
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df_regime, x='pollution_regime', y='coverage_90pct', hue='pollution_regime', palette='YlOrRd', legend=False, ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Empirical Coverage across Pollution Regimes (Nominal 90%)")
        ax.set_xlabel("Pollution Regime")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.5, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "coverage_by_regime.png", dpi=150)
        plt.close(fig)

        # 6. Interval Width by Regime
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df_regime, x='pollution_regime', y='mean_width_90pct_ugm3', hue='pollution_regime', palette='crest', legend=False, ax=ax)
        ax.set_title("Mean Prediction Interval Width (MPIW) by Pollution Regime")
        ax.set_xlabel("Pollution Regime")
        ax.set_ylabel("90% MPIW (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "interval_width_by_regime.png", dpi=150)
        plt.close(fig)

        # 7. Uncertainty by Season
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df_seasonal, x='season', y='mean_spread_ugm3', hue='season', palette='Set2', legend=False, ax=ax)
        ax.set_title("Mean Ensemble Spread across Seasons")
        ax.set_xlabel("Season")
        ax.set_ylabel("Ensemble Spread σ (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "uncertainty_by_season.png", dpi=150)
        plt.close(fig)

        # 8. Uncertainty by Year
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=self.df_yearly, x='year', y='spearman_spread_error_corr', hue='year', palette='Pastel1', legend=False, ax=ax)
        ax.set_title("Spread-Error Correlation Stability across Evaluation Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Spearman Rank Correlation (ρ)")
        plt.tight_layout()
        fig.savefig(plot_dir / "uncertainty_by_year.png", dpi=150)
        plt.close(fig)

        # 9. Extreme Pollution Uncertainty
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df_extreme, x='threshold_category', y='coverage_90pct', hue='threshold_category', palette='Reds_r', legend=False, ax=ax)
        ax.axhline(0.90, color="blue", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Empirical Coverage during Extreme Episodes")
        ax.set_xlabel("Episode Threshold")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.5, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "extreme_pollution_uncertainty.png", dpi=150)
        plt.close(fig)

        # 10. Bootstrap vs Seed Spread Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(self.df_boot['ensemble_std'], color="teal", fill=True, label="Bootstrap Ensemble Spread (B=30)", ax=ax)
        sns.kdeplot(self.df_seed['ensemble_std'], color="darkorange", fill=True, label="Random-Seed Ensemble Spread (N=30)", ax=ax)
        ax.set_title("Ensemble Spread Density: Bootstrap vs. Random-Seed")
        ax.set_xlabel("Ensemble Spread σ (µg/m³)")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "bootstrap_vs_seed_spread.png", dpi=150)
        plt.close(fig)

        # 11. Ensemble Size Stability
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.df_sens['ensemble_size'], self.df_sens['mean_spread_ugm3'], marker='o', color="purple", label="Mean Spread (µg/m³)")
        ax.plot(self.df_sens['ensemble_size'], self.df_sens['prediction_mae_ugm3'], marker='s', color="forestgreen", label="Prediction MAE (µg/m³)")
        ax.set_title("Ensemble Sensitivity: Metric Stability vs. Ensemble Size")
        ax.set_xlabel("Ensemble Size (N)")
        ax.set_ylabel("Metric Value (µg/m³)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "ensemble_size_stability.png", dpi=150)
        plt.close(fig)

        # 12. Representative Prediction Intervals Time Series
        sub_rep = self.df_boot[
            (self.df_boot['date_dt'] >= '2024-10-15') &
            (self.df_boot['date_dt'] <= '2024-12-15')
        ].sort_values('date_dt')

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sub_rep['date_dt'], sub_rep['observed_pm25'], color="black", marker='o', markersize=3, label="Observed PM2.5")
        ax.plot(sub_rep['date_dt'], sub_rep['ensemble_mean'], color="blue", linestyle="--", label="Bootstrap Ensemble Mean")
        ax.fill_between(sub_rep['date_dt'], np.maximum(0.0, sub_rep['q05']), sub_rep['q95'], color="cornflowerblue", alpha=0.35, label="90% Empirical Ensemble Interval")
        ax.set_title("Representative Ensemble Prediction Intervals (Stubble & Inversion Season 2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig.savefig(plot_dir / "prediction_intervals_representative_cases.png", dpi=150)
        plt.close(fig)

        # 13. Ensemble Error Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.df_boot['residual'], kde=True, color="darkcyan", bins=35, ax=ax)
        ax.set_title("Ensemble Mean Residual Distribution (2022–2024)")
        ax.set_xlabel("Residual (Observed - Ensemble Mean) (µg/m³)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(plot_dir / "ensemble_error_distribution.png", dpi=150)
        plt.close(fig)

        # 14. Uncertainty Discrimination (ROC / PR Curves)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df_disc, x='error_threshold_definition', y='roc_auc', hue='error_threshold_definition', palette='mako', legend=False, ax=ax)
        ax.axhline(0.50, color="gray", linestyle="--", label="Random Discrimination (0.50)")
        ax.set_title("Uncertainty Discrimination (ROC-AUC for High-Error Detection)")
        ax.set_xlabel("High-Error Threshold")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.4, 1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "uncertainty_discrimination.png", dpi=150)
        plt.close(fig)

        logger.info("All 14 publication figures generated cleanly.")
