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

logger = setup_logger("VisualizationPhase6C")


class VisualizationEnginePhase6C:
    """
    Publication-Quality Visualization Generator for Phase 6C.
    Generates 12 diagnostic figures under ml/experiments/phase6c/plots/.
    """

    def __init__(
        self,
        df_bench: pd.DataFrame,
        df_intervals: pd.DataFrame,
        df_regime: pd.DataFrame,
        df_seasonal: pd.DataFrame,
        df_yearly: pd.DataFrame,
        df_extreme: pd.DataFrame,
        best_method: str
    ):
        self.df_bench = df_bench.copy()
        self.df_intervals = df_intervals.copy()
        self.df_intervals['date_dt'] = pd.to_datetime(self.df_intervals['date'])
        self.df_regime = df_regime.copy()
        self.df_seasonal = df_seasonal.copy()
        self.df_yearly = df_yearly.copy()
        self.df_extreme = df_extreme.copy()
        self.best_method = best_method

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

        # 1. Conformal Coverage Comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        sub_90 = self.df_bench[self.df_bench['nominal_coverage'] == 0.90]
        sns.barplot(data=sub_90, x='method', y='empirical_coverage', hue='method', palette='Blues_r', legend=False, ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Empirical Coverage Comparison across Methods (Nominal 90%)")
        ax.set_xlabel("Uncertainty Method")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.2, 1.05)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "1_conformal_coverage_comparison.png", dpi=150)
        plt.close(fig)

        # 2. Interval Width Comparison (MPIW)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sub_90, x='method', y='mean_width_ugm3', hue='method', palette='crest', legend=False, ax=ax)
        ax.set_title("Mean Prediction Interval Width (MPIW) across Methods (Nominal 90%)")
        ax.set_xlabel("Uncertainty Method")
        ax.set_ylabel("MPIW (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(plot_dir / "2_interval_width_comparison.png", dpi=150)
        plt.close(fig)

        # 3. Calibration Curve
        fig, ax = plt.subplots(figsize=(7, 7))
        nom_levels = [0.80, 0.90, 0.95]
        ax.plot(nom_levels, nom_levels, color="gray", linestyle="--", label="Ideal Calibration")

        for m_name, col in [
            ("standard_conformal", "blue"),
            ("normalized_conformal", "darkorange"),
            ("ensemble_regime_conformal_hybrid", "forestgreen")
        ]:
            sub_m = self.df_bench[self.df_bench['method'] == m_name]
            if not sub_m.empty:
                emp_covs = [float(sub_m[sub_m['nominal_coverage'] == n]['empirical_coverage'].iloc[0]) for n in nom_levels]
                ax.plot(nom_levels, emp_covs, marker='o', linewidth=2, color=col, label=m_name)

        ax.set_title("Calibration Curve across Conformal Methods")
        ax.set_xlabel("Nominal Coverage Level")
        ax.set_ylabel("Empirical Coverage Level")
        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(0.70, 1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "3_calibration_curve.png", dpi=150)
        plt.close(fig)

        # 4. Coverage by Pollution Regime
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=self.df_regime, x='pollution_regime', y='coverage_90pct', hue='method', palette='tab10', ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Coverage across Pollution Regimes by Conformal Method (Nominal 90%)")
        ax.set_xlabel("Pollution Regime")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.6, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "4_coverage_by_pollution_regime.png", dpi=150)
        plt.close(fig)

        # 5. Coverage by Season
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=self.df_seasonal, x='season', y='coverage_90pct', hue='method', palette='tab10', ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Coverage across Seasons by Conformal Method (Nominal 90%)")
        ax.set_xlabel("Season")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.6, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "5_coverage_by_season.png", dpi=150)
        plt.close(fig)

        # 6. Coverage by Year
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df_yearly, x='year', y='coverage_90pct', hue='method', palette='tab10', ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Year-to-Year Coverage Stability (2022–2024)")
        ax.set_xlabel("Evaluation Year")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.75, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "6_coverage_by_year.png", dpi=150)
        plt.close(fig)

        # 7. Extreme Event Coverage
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=self.df_extreme, x='threshold_definition', y='coverage_90pct', hue='method', palette='tab10', ax=ax)
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Extreme Episode Coverage Performance (PM2.5 >= 150 & >= 250 µg/m³)")
        ax.set_xlabel("Episode Threshold")
        ax.set_ylabel("Empirical Coverage")
        ax.set_ylim(0.6, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "7_extreme_event_coverage.png", dpi=150)
        plt.close(fig)

        # 8. Winkler Score Comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sub_90, x='method', y='winkler_interval_score', hue='method', palette='mako', legend=False, ax=ax)
        ax.set_title("Winkler Interval Score Comparison (Lower is Better, Nominal 90%)")
        ax.set_xlabel("Method")
        ax.set_ylabel("Winkler Score")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        fig.savefig(plot_dir / "8_winkler_score_comparison.png", dpi=150)
        plt.close(fig)

        # 9. Interval Width vs. Coverage Tradeoff
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=sub_90, x='mean_width_ugm3', y='empirical_coverage', hue='method', s=120, palette='tab10', ax=ax)
        ax.axhline(0.90, color="red", linestyle="--", label="Nominal 90% Target")
        ax.set_title("Efficiency Tradeoff: Interval Width (MPIW) vs. Empirical Coverage")
        ax.set_xlabel("MPIW (µg/m³)")
        ax.set_ylabel("Empirical Coverage")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "9_interval_width_vs_coverage.png", dpi=150)
        plt.close(fig)

        # 10. Ensemble vs Conformal Intervals Comparison
        sub_comp = self.df_intervals[
            (self.df_intervals['method'].isin(['standard_conformal', 'ensemble_regime_conformal_hybrid'])) &
            (self.df_intervals['nominal_coverage'] == 0.90) &
            (self.df_intervals['date_dt'] >= '2024-11-01') &
            (self.df_intervals['date_dt'] <= '2024-12-01')
        ].sort_values('date_dt')

        fig, ax = plt.subplots(figsize=(12, 5))
        sub_hyb = sub_comp[sub_comp['method'] == 'ensemble_regime_conformal_hybrid']
        sub_std = sub_comp[sub_comp['method'] == 'standard_conformal']

        ax.plot(sub_hyb['date_dt'], sub_hyb['observed_pm25'], color="black", marker='o', markersize=3, label="Observed PM2.5")
        ax.fill_between(sub_std['date_dt'], sub_std['lower_bound'], sub_std['upper_bound'], color="gray", alpha=0.25, label="Standard Conformal (Fixed Width)")
        ax.fill_between(sub_hyb['date_dt'], sub_hyb['lower_bound'], sub_hyb['upper_bound'], color="green", alpha=0.35, label="Hybrid Conformal (Adaptive Width)")
        ax.set_title("Standard vs. Adaptive Hybrid Conformal Prediction Intervals (Nov 2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig.savefig(plot_dir / "10_ensemble_vs_conformal_intervals.png", dpi=150)
        plt.close(fig)

        # 11. Representative Conformal Cases
        sub_rep = self.df_intervals[
            (self.df_intervals['method'] == self.best_method) &
            (self.df_intervals['nominal_coverage'] == 0.90) &
            (self.df_intervals['date_dt'] >= '2024-10-15') &
            (self.df_intervals['date_dt'] <= '2024-12-15')
        ].sort_values('date_dt')

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sub_rep['date_dt'], sub_rep['observed_pm25'], color="black", marker='o', markersize=3, label="Observed PM2.5")
        ax.fill_between(sub_rep['date_dt'], sub_rep['lower_bound'], sub_rep['upper_bound'], color="dodgerblue", alpha=0.35, label=f"90% {self.best_method}")
        ax.set_title(f"Representative Conformal Prediction Intervals ({self.best_method}, Oct-Dec 2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        fig.savefig(plot_dir / "11_representative_conformal_cases.png", dpi=150)
        plt.close(fig)

        # 12. Temporal Coverage Stability
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sub_best_all = self.df_intervals[
            (self.df_intervals['method'] == self.best_method) &
            (self.df_intervals['nominal_coverage'] == 0.90)
        ].sort_values('date_dt')
        sub_best_all['rolling_cov_30d'] = sub_best_all['covered'].astype(float).rolling(30, min_periods=10).mean()

        ax.plot(sub_best_all['date_dt'], sub_best_all['rolling_cov_30d'], color="teal", linewidth=2, label="30-Day Rolling Empirical Coverage")
        ax.axhline(0.90, color="crimson", linestyle="--", label="Nominal 90% Target")
        ax.set_title(f"30-Day Rolling Conformal Coverage Stability ({self.best_method}, 2022–2024)")
        ax.set_xlabel("Date")
        ax.set_ylabel("30-Day Empirical Coverage")
        ax.set_ylim(0.7, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "12_temporal_coverage_stability.png", dpi=150)
        plt.close(fig)

        logger.info("All 12 publication figures generated cleanly.")
