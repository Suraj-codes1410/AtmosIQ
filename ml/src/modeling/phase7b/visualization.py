"""
AtmosIQ Phase 7B: Publication Visualization Engine.
"""

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class VisualizationEnginePhase7B:
    """Generates 12 publication-quality plots comparing Observed vs Synthetic distributions."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.titlesize': 14
        })

    def generate_all_plots(
        self,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        df_audit: pd.DataFrame,
        regime_trans_mat: np.ndarray,
        plot_dir: Path
    ):
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Observed vs Synthetic PM2.5 Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(df_real['pm25'], label='Observed (2020-2021)', color='navy', fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(df_synthetic['pm25'], label='Synthetic (HP-STG)', color='crimson', fill=True, alpha=0.3, ax=ax)
        ax.set_title("Observed vs Synthetic PM2.5 Density Distribution")
        ax.set_xlabel("PM2.5 (µg/m³)")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "observed_vs_synthetic_pm25_distribution.png", dpi=150)
        plt.close(fig)

        # 2. Observed vs Synthetic PM2.5 Timeseries
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_real['pm25'].values[:120], label='Observed (First 120 Days)', color='navy', alpha=0.8, lw=1.5)
        ax.plot(df_synthetic['pm25'].values[:120], label='Synthetic (First 120 Days)', color='crimson', alpha=0.8, lw=1.5, ls='--')
        ax.set_title("PM2.5 Trajectory Comparison (120-Day Window)")
        ax.set_xlabel("Relative Time Step (Days)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "observed_vs_synthetic_pm25_timeseries.png", dpi=150)
        plt.close(fig)

        # 3. Observed vs Synthetic ACF
        def compute_acf(s, max_lag=14):
            s = s - np.mean(s)
            var = np.var(s)
            if var == 0: return np.ones(max_lag)
            return [np.corrcoef(s[:-k], s[k:])[0, 1] for k in range(1, max_lag + 1)]

        acf_r = compute_acf(df_real['pm25'].dropna().values, 14)
        acf_s = compute_acf(df_synthetic['pm25'].dropna().values, 14)
        lags = np.arange(1, 15)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(lags, acf_r, marker='o', label='Observed ACF', color='navy', lw=2)
        ax.plot(lags, acf_s, marker='s', label='Synthetic ACF', color='crimson', lw=2, ls='--')
        ax.set_title("Autocorrelation Function (ACF) Lags 1–14")
        ax.set_xlabel("Lag (Days)")
        ax.set_ylabel("Autocorrelation")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "observed_vs_synthetic_acf.png", dpi=150)
        plt.close(fig)

        # 4. Observed vs Synthetic Correlation Heatmap
        eval_feats = ['pm25', 'pm25_roll_mean_3d', 'temperature_c_lag_1d', 'humidity_pct_lag_1d', 'wind_speed_kmh_lag_1d', 'pblh_1d', 'ventilation_index_1d', 'aod_550_1d']
        eval_feats = [f for f in eval_feats if f in df_real.columns and f in df_synthetic.columns]

        corr_r = df_real[eval_feats].corr()
        corr_s = df_synthetic[eval_feats].corr()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.heatmap(corr_r, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, ax=axes[0])
        axes[0].set_title("Observed Feature Correlation")
        sns.heatmap(corr_s, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1])
        axes[1].set_title("Synthetic Feature Correlation")
        plt.tight_layout()
        fig.savefig(plot_dir / "observed_vs_synthetic_correlation_heatmap.png", dpi=150)
        plt.close(fig)

        # 5. Observed vs Synthetic Regime Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        r_counts = df_real['pollution_regime'].value_counts(normalize=True).reindex(['Low', 'Moderate', 'High', 'Extreme']).fillna(0) * 100
        s_counts = df_synthetic['pollution_regime'].value_counts(normalize=True).reindex(['Low', 'Moderate', 'High', 'Extreme']).fillna(0) * 100
        df_bars = pd.DataFrame({'Observed': r_counts, 'Synthetic': s_counts})
        df_bars.plot(kind='bar', color=['navy', 'crimson'], ax=ax)
        ax.set_title("Pollution Regime Proportions (% of Days)")
        ax.set_xlabel("Regime")
        ax.set_ylabel("Percentage (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(plot_dir / "observed_vs_synthetic_regime_distribution.png", dpi=150)
        plt.close(fig)

        # 6. Observed vs Synthetic Season Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        r_seas = df_real['season'].value_counts(normalize=True).reindex(['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']).fillna(0) * 100
        s_seas = df_synthetic['season'].value_counts(normalize=True).reindex(['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']).fillna(0) * 100
        df_s_bars = pd.DataFrame({'Observed': r_seas, 'Synthetic': s_seas})
        df_s_bars.plot(kind='bar', color=['navy', 'crimson'], ax=ax)
        ax.set_title("Seasonal Distribution Comparison (% of Days)")
        ax.set_xlabel("Season")
        ax.set_ylabel("Percentage (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(plot_dir / "observed_vs_synthetic_season_distribution.png", dpi=150)
        plt.close(fig)

        # 7. Extreme Event Coherence
        fig, ax = plt.subplots(figsize=(8, 5))
        df_ext = df_synthetic[df_synthetic['pm25'] >= 250.0]
        if len(df_ext) > 0:
            sns.scatterplot(data=df_ext, x='ventilation_index_1d', y='pm25', hue='season', palette='Set1', ax=ax, s=60)
            ax.axvline(4500.0, color='gray', ls='--', label='Max Coherent VI')
            ax.set_title("Synthetic Extreme Events (PM2.5 ≥ 250 µg/m³) vs Ventilation Index")
            ax.set_xlabel("Ventilation Index (m²/s)")
            ax.set_ylabel("PM2.5 (µg/m³)")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No Extreme Events Generated", ha='center', va='center')
        plt.tight_layout()
        fig.savefig(plot_dir / "extreme_event_coherence.png", dpi=150)
        plt.close(fig)

        # 8. Trajectory Examples
        fig, ax = plt.subplots(figsize=(12, 5))
        sample_trajs = df_synthetic['trajectory_id'].unique()[:4]
        for tid in sample_trajs:
            sub = df_synthetic[df_synthetic['trajectory_id'] == tid]
            ax.plot(sub['step_idx'], sub['pm25'], label=f"Trajectory {tid}", lw=1.8)
        ax.set_title("Representative Synthetic PM2.5 Trajectory Profiles")
        ax.set_xlabel("Step Index (Days)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "trajectory_examples.png", dpi=150)
        plt.close(fig)

        # 9. Physics Constraint Corrections
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if len(df_audit) > 0:
            c_counts = df_audit['constraint_name'].value_counts()
            c_counts.plot(kind='barh', color='teal', ax=ax)
            ax.set_title("Physics Constraint Correction Counts")
            ax.set_xlabel("Correction Frequency")
        else:
            ax.text(0.5, 0.5, "Zero Corrections (100% Unconstrained Physical Pass)", ha='center', va='center')
            ax.set_title("Physics Constraint Audit")
        plt.tight_layout()
        fig.savefig(plot_dir / "physics_constraint_corrections.png", dpi=150)
        plt.close(fig)

        # 10. Synthetic PM2.5 Extreme Tail
        fig, ax = plt.subplots(figsize=(8, 5))
        q_r = np.percentile(df_real['pm25'].dropna(), np.linspace(80, 99.5, 50))
        q_s = np.percentile(df_synthetic['pm25'].dropna(), np.linspace(80, 99.5, 50))
        ax.scatter(q_r, q_s, color='crimson', s=40, label='Upper Tail Quantiles (Q80-Q99.5)')
        min_v, max_v = min(q_r.min(), q_s.min()), max(q_r.max(), q_s.max())
        ax.plot([min_v, max_v], [min_v, max_v], color='navy', ls='--', label='1:1 Line')
        ax.set_title("Quantile-Quantile (Q-Q) Upper Tail Heavy PM2.5 Comparison")
        ax.set_xlabel("Observed PM2.5 (µg/m³)")
        ax.set_ylabel("Synthetic PM2.5 (µg/m³)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "synthetic_pm25_extreme_tail.png", dpi=150)
        plt.close(fig)

        # 11. Observed vs Synthetic Feature Distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        f_list = ['temperature_c_lag_1d', 'humidity_pct_lag_1d', 'wind_speed_kmh_lag_1d', 'pblh_1d']
        for idx, f in enumerate(f_list):
            r, c = idx // 2, idx % 2
            ax = axes[r, c]
            sns.kdeplot(df_real[f], label='Observed', color='navy', fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(df_synthetic[f], label='Synthetic', color='crimson', fill=True, alpha=0.3, ax=ax)
            ax.set_title(f)
            ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "observed_vs_synthetic_feature_distributions.png", dpi=150)
        plt.close(fig)

        # 12. Regime Transition Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        reg_names = ['Low', 'Moderate', 'High', 'Extreme']
        sns.heatmap(regime_trans_mat, annot=True, fmt='.2f', cmap='Blues', xticklabels=reg_names, yticklabels=reg_names, ax=ax)
        ax.set_title("Learned Regime Markov Transition Probabilities")
        ax.set_xlabel("Next Regime (t+1)")
        ax.set_ylabel("Current Regime (t)")
        plt.tight_layout()
        fig.savefig(plot_dir / "regime_transition_matrix.png", dpi=150)
        plt.close(fig)
