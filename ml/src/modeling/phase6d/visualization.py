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

logger = setup_logger("VisualizationPhase6D")


class VisualizationEnginePhase6D:
    """
    Publication-Quality Diagnostic Visualization Engine for Phase 6D.
    Generates 12 figures under ml/experiments/phase6d/plots/.
    """

    def __init__(
        self,
        df_bench: pd.DataFrame,
        df_intervals: pd.DataFrame,
        df_temp_stab: pd.DataFrame,
        df_extreme: pd.DataFrame,
        df_unif: pd.DataFrame,
        df_worst: pd.DataFrame,
        df_evol: pd.DataFrame
    ):
        self.df_bench = df_bench.copy()
        self.df_intervals = df_intervals.copy()
        self.df_intervals['date_dt'] = pd.to_datetime(self.df_intervals['date'])
        self.df_temp_stab = df_temp_stab.copy()
        self.df_extreme = df_extreme.copy()
        self.df_unif = df_unif.copy()
        self.df_worst = df_worst.copy()
        self.df_evol = df_evol.copy()

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
        logger.info(f"Generating 12 publication-quality figures in {plot_dir}...")
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Final Coverage Comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        sub_90 = self.df_bench[self.df_bench['nominal_coverage'] == 0.90]
        sns.barplot(data=sub_90, x='method', y='empirical_coverage', hue='method', palette='Blues_r', legend=False, ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Final Empirical Coverage Comparison across Methods (Nominal 90%)")
        ax.set_xlabel("Method")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.2, 1.05)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_coverage_comparison.png", dpi=150)
        plt.close(fig)

        # 2. Final Calibration Curve
        fig, ax = plt.subplots(figsize=(7, 7))
        nom_levels = [0.80, 0.90, 0.95]
        ax.plot(nom_levels, nom_levels, color="gray", linestyle="--", label="Ideal Calibration")

        sub_norm = self.df_bench[self.df_bench['method'] == 'normalized_conformal']
        if not sub_norm.empty:
            emp_norm = [float(sub_norm[sub_norm['nominal_coverage'] == n]['empirical_coverage'].iloc[0]) for n in nom_levels]
            ax.plot(nom_levels, emp_norm, marker='o', linewidth=2.5, color="teal", label="Normalized Conformal (Production Selected)")

        sub_std = self.df_bench[self.df_bench['method'] == 'standard_conformal']
        if not sub_std.empty:
            emp_std = [float(sub_std[sub_std['nominal_coverage'] == n]['empirical_coverage'].iloc[0]) for n in nom_levels]
            ax.plot(nom_levels, emp_std, marker='s', linewidth=1.5, color="gray", linestyle=":", label="Standard Conformal")

        ax.set_title("Final Production Prediction Interval Calibration Curve")
        ax.set_xlabel("Nominal Coverage Level")
        ax.set_ylabel("Empirical Coverage Level")
        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(0.70, 1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_calibration_curve.png", dpi=150)
        plt.close(fig)

        # 3. Final Interval Width Comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sub_90, x='method', y='mean_width_ugm3', hue='method', palette='mako', legend=False, ax=ax)
        ax.set_title("Mean Prediction Interval Width (MPIW) across Methods (Nominal 90%)")
        ax.set_xlabel("Method")
        ax.set_ylabel("MPIW (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(plot_dir / "final_interval_width_comparison.png", dpi=150)
        plt.close(fig)

        # 4. Final Coverage by Regime
        sub_reg = self.df_unif[self.df_unif['slice_category'] == 'Pollution Regime']
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=sub_reg, x='slice_name', y='empirical_coverage_90pct', hue='slice_name', palette='YlOrRd', legend=False, ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Normalized Conformal Coverage across Pollution Regimes")
        ax.set_xlabel("Pollution Regime")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.75, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_coverage_by_regime.png", dpi=150)
        plt.close(fig)

        # 5. Final Coverage by Season
        sub_seas = self.df_unif[self.df_unif['slice_category'] == 'Season']
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=sub_seas, x='slice_name', y='empirical_coverage_90pct', hue='slice_name', palette='Set2', legend=False, ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Normalized Conformal Coverage across Seasons")
        ax.set_xlabel("Season")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.75, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_coverage_by_season.png", dpi=150)
        plt.close(fig)

        # 6. Final Coverage by Year
        sub_yr = self.df_unif[self.df_unif['slice_category'] == 'Evaluation Year']
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=sub_yr, x='slice_name', y='empirical_coverage_90pct', hue='slice_name', palette='Blues', legend=False, ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Normalized Conformal Annual Coverage Stability (2022–2024)")
        ax.set_xlabel("Evaluation Year")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.80, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_coverage_by_year.png", dpi=150)
        plt.close(fig)

        # 7. Final Extreme Threshold Stress Test
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(self.df_extreme['threshold_ugm3'], self.df_extreme['empirical_coverage_90pct'], marker='o', linewidth=2.5, color="firebrick", label="90% Empirical Coverage")
        ax.axhline(0.90, color="navy", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Extreme Pollution Stress Test: Coverage vs. Severity Threshold")
        ax.set_xlabel("Pollution Threshold (µg/m³)")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.75, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_extreme_threshold_stress_test.png", dpi=150)
        plt.close(fig)

        # 8. Final Temporal Rolling Coverage
        sub_best = self.df_intervals[self.df_intervals['nominal_coverage'] == 0.90].sort_values('date_dt')
        sub_best['roll_30d'] = sub_best['covered'].astype(float).rolling(30, min_periods=10).mean()
        sub_best['roll_60d'] = sub_best['covered'].astype(float).rolling(60, min_periods=20).mean()

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(sub_best['date_dt'], sub_best['roll_30d'], color="teal", alpha=0.7, label="30-Day Rolling Coverage")
        ax.plot(sub_best['date_dt'], sub_best['roll_60d'], color="darkblue", linewidth=2, label="60-Day Rolling Coverage")
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Normalized Conformal Rolling Temporal Coverage Stability (2022–2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.70, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_temporal_rolling_coverage.png", dpi=150)
        plt.close(fig)

        # 9. Final Coverage vs Width Tradeoff
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=sub_90, x='mean_width_ugm3', y='empirical_coverage', hue='method', s=140, palette='tab10', ax=ax)
        ax.axhline(0.90, color="red", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Efficiency Tradeoff: Mean Interval Width vs. Empirical Coverage")
        ax.set_xlabel("MPIW (µg/m³)")
        ax.set_ylabel("Empirical Coverage")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "final_coverage_vs_width.png", dpi=150)
        plt.close(fig)

        # 10. Worst-Case Miscoverage
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=self.df_worst.head(10), x='date', y='violation_magnitude_ugm3', hue='violation_type', palette='Reds_r', ax=ax)
        ax.set_title("Top 10 Worst-Case Prediction Bound Violations")
        ax.set_xlabel("Event Date")
        ax.set_ylabel("Miscoverage Breach (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(plot_dir / "final_worst_case_miscoverage.png", dpi=150)
        plt.close(fig)

        # 11. Final Prediction Intervals Extreme Events
        sub_rep = self.df_intervals[
            (self.df_intervals['nominal_coverage'] == 0.90) &
            (self.df_intervals['date_dt'] >= '2024-10-15') &
            (self.df_intervals['date_dt'] <= '2024-12-15')
        ].sort_values('date_dt')

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sub_rep['date_dt'], sub_rep['observed_pm25'], color="black", marker='o', markersize=3, label="Observed PM2.5")
        ax.fill_between(sub_rep['date_dt'], sub_rep['lower_bound'], sub_rep['upper_bound'], color="cornflowerblue", alpha=0.4, label="Normalized Conformal (90% Interval)")
        ax.set_title("Validated Production Prediction Intervals during Peak Stubble & Inversion (2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig.savefig(plot_dir / "final_prediction_intervals_extreme_events.png", dpi=150)
        plt.close(fig)

        # 12. Final Uncertainty Evolution
        fig, ax = plt.subplots(figsize=(10, 4.5))
        phases = ["Phase 6A (Baseline)", "Phase 6B (Raw Ens)", "Phase 6C (Conformal)", "Phase 6D (Validated)"]
        covs = [91.45, 29.29, 89.78, 89.78]
        colors = ["lightgray", "coral", "mediumseagreen", "teal"]
        bars = ax.bar(phases, covs, color=colors, width=0.55)
        ax.axhline(90.0, color="crimson", linestyle="--", label="Nominal 90% Target")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        ax.set_title("Evolution of 90% Prediction Interval Coverage Across Phase 6")
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_ylim(0, 110)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "final_uncertainty_evolution.png", dpi=150)
        plt.close(fig)

        logger.info("All 12 publication figures generated cleanly.")
