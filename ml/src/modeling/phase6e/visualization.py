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

logger = setup_logger("VisualizationPhase6E")


class VisualizationEnginePhase6E:
    """
    Publication-Quality Diagnostic Visualization Generator for Phase 6E.
    Generates 12 figures under ml/experiments/phase6e/plots/.
    """

    def __init__(
        self,
        df_shap_obs: pd.DataFrame,
        df_feat_summary: pd.DataFrame,
        df_grp_obs: pd.DataFrame,
        df_grp_summary: pd.DataFrame,
        df_cf: pd.DataFrame,
        df_sc_summary: pd.DataFrame,
        df_ood: pd.DataFrame
    ):
        self.df_shap_obs = df_shap_obs.copy()
        self.df_feat_summary = df_feat_summary.copy()
        self.df_grp_obs = df_grp_obs.copy()
        self.df_grp_summary = df_grp_summary.copy()
        self.df_cf = df_cf.copy()
        self.df_sc_summary = df_sc_summary.copy()
        self.df_ood = df_ood.copy()

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

        # 1. SHAP Attribution Uncertainty Ranking
        top_15 = self.df_feat_summary.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_15, x='mean_absolute_shap', y='feature_name', hue='stability_classification', palette={'HIGH_STABILITY': 'teal', 'MODERATE_STABILITY': 'darkorange', 'LOW_STABILITY': 'crimson'}, ax=ax)
        ax.set_title("Top 15 Features by Mean Absolute SHAP & Attribution Stability")
        ax.set_xlabel("Mean Absolute SHAP Magnitude (µg/m³)")
        ax.set_ylabel("Feature Name")
        plt.tight_layout()
        fig.savefig(plot_dir / "1_shap_attribution_uncertainty_ranking.png", dpi=150)
        plt.close(fig)

        # 2. Mean SHAP with 90% Attribution Intervals
        fig, ax = plt.subplots(figsize=(10, 6))
        top_10 = self.df_feat_summary.head(10)['feature_name'].tolist()
        sub_top10 = self.df_shap_obs[self.df_shap_obs['feature_name'].isin(top_10)].groupby('feature_name').agg({
            'mean_shap': 'mean',
            'q05_shap': 'mean',
            'q95_shap': 'mean'
        }).loc[top_10].reset_index()

        y_pos = np.arange(len(sub_top10))
        ax.errorbar(
            sub_top10['mean_shap'], y_pos,
            xerr=[sub_top10['mean_shap'] - sub_top10['q05_shap'], sub_top10['q95_shap'] - sub_top10['mean_shap']],
            fmt='o', color='navy', ecolor='cornflowerblue', elinewidth=2, capsize=4
        )
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_top10['feature_name'])
        ax.set_title("Top 10 Features: Mean SHAP with 90% Attribution Intervals")
        ax.set_xlabel("Attribution Value (µg/m³)")
        ax.set_ylabel("Feature Name")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(plot_dir / "2_mean_shap_with_90pct_intervals.png", dpi=150)
        plt.close(fig)

        # 3. SHAP Sign Stability by Feature
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_15, x='mean_sign_stability', y='feature_name', palette='crest', ax=ax)
        ax.axvline(0.90, color='crimson', linestyle='--', label='High Stability Threshold (90%)')
        ax.set_title("Attribution Sign Stability across Top 15 Features")
        ax.set_xlabel("Sign Stability Fraction (max(P(>0), P(<0)))")
        ax.set_ylabel("Feature Name")
        ax.set_xlim(0.5, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "3_shap_sign_stability_by_feature.png", dpi=150)
        plt.close(fig)

        # 4. Group Attribution Uncertainty
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=self.df_grp_summary, x='mean_absolute_group_shap', y='feature_group', palette='mako', ax=ax)
        ax.set_title("Environmental Group Attribution Magnitude (Mean Absolute Group SHAP)")
        ax.set_xlabel("Mean Absolute Group SHAP (µg/m³)")
        ax.set_ylabel("Environmental Process Group")
        plt.tight_layout()
        fig.savefig(plot_dir / "4_group_attribution_uncertainty.png", dpi=150)
        plt.close(fig)

        # 5. Counterfactual Delta Distributions
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=self.df_cf, x='scenario_name', y='mean_delta_pm25', palette='Set3', ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title("Counterfactual ΔPM2.5 Distributions across Scenarios")
        ax.set_xlabel("Scenario Name")
        ax.set_ylabel("ΔPM2.5 Response (µg/m³)")
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        fig.savefig(plot_dir / "5_counterfactual_delta_distributions.png", dpi=150)
        plt.close(fig)

        # 6. Counterfactual Uncertainty Intervals
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(self.df_sc_summary))
        ax.errorbar(
            self.df_sc_summary['mean_delta_pm25'], y_pos,
            xerr=[self.df_sc_summary['mean_delta_pm25'] - self.df_sc_summary['q10_delta_mean'], self.df_sc_summary['q90_delta_mean'] - self.df_sc_summary['mean_delta_pm25']],
            fmt='s', color='darkgreen', ecolor='lightgreen', elinewidth=2.5, capsize=4
        )
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.df_sc_summary['scenario_name'])
        ax.set_title("Counterfactual Expected ΔPM2.5 with 80% Uncertainty Intervals")
        ax.set_xlabel("Expected ΔPM2.5 (µg/m³)")
        ax.set_ylabel("Scenario Name")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(plot_dir / "6_counterfactual_uncertainty_intervals.png", dpi=150)
        plt.close(fig)

        # 7. OOD Score vs Prediction Uncertainty
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=self.df_ood.sample(min(len(self.df_ood), 1000), random_state=42), x='ood_score', y='cf_delta_std', hue='scenario_name', alpha=0.6, s=40, ax=ax)
        ax.set_title("OOD Score vs. Counterfactual Response Dispersion")
        ax.set_xlabel("OOD Standardized Distance Score")
        ax.set_ylabel("Counterfactual ΔPM2.5 Std Dev (µg/m³)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "7_ood_score_vs_prediction_uncertainty.png", dpi=150)
        plt.close(fig)

        # 8. OOD Score vs Counterfactual Interval Width
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=self.df_ood.sample(min(len(self.df_ood), 1000), random_state=42), x='ood_score', y='cf_interval_width_90', hue='ood_status', palette='tab10', alpha=0.6, s=40, ax=ax)
        ax.set_title("OOD Shift vs. 90% Counterfactual Interval Width")
        ax.set_xlabel("OOD Standardized Distance Score")
        ax.set_ylabel("90% Interval Width (µg/m³)")
        plt.tight_layout()
        fig.savefig(plot_dir / "8_ood_score_vs_counterfactual_uncertainty.png", dpi=150)
        plt.close(fig)

        # 9. Attribution Uncertainty Across Pollution Regimes
        reg_grp = self.df_grp_obs.groupby(['pollution_regime', 'feature_group'])['mean_group_shap'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=reg_grp, x='pollution_regime', y='mean_group_shap', hue='feature_group', palette='tab10', ax=ax)
        ax.set_title("Environmental Group Attribution across Pollution Regimes")
        ax.set_xlabel("Pollution Regime")
        ax.set_ylabel("Signed Mean Group SHAP (µg/m³)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "9_attribution_uncertainty_across_regimes.png", dpi=150)
        plt.close(fig)

        # 10. Attribution Uncertainty Across Seasons
        seas_grp = self.df_grp_obs.groupby(['season', 'feature_group'])['mean_group_shap'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=seas_grp, x='season', y='mean_group_shap', hue='feature_group', palette='tab10', ax=ax)
        ax.set_title("Environmental Group Attribution across Seasons")
        ax.set_xlabel("Season")
        ax.set_ylabel("Signed Mean Group SHAP (µg/m³)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(plot_dir / "10_attribution_uncertainty_across_seasons.png", dpi=150)
        plt.close(fig)

        # 11. Counterfactual Directional Stability
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=self.df_sc_summary, x='mean_directional_stability', y='scenario_name', palette='Blues_r', ax=ax)
        ax.axvline(0.90, color='crimson', linestyle='--', label='90% Directional Stability')
        ax.set_title("Directional Stability Fraction across Counterfactual Scenarios")
        ax.set_xlabel("Directional Stability (fraction sign(Δ) == sign(mean(Δ)))")
        ax.set_ylabel("Scenario Name")
        ax.set_xlim(0.5, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "11_counterfactual_directional_stability.png", dpi=150)
        plt.close(fig)

        # 12. Feature Attribution Stability vs Magnitude
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=self.df_feat_summary, x='mean_absolute_shap', y='mean_sign_stability', hue='stability_classification', s=120, palette={'HIGH_STABILITY': 'teal', 'MODERATE_STABILITY': 'darkorange', 'LOW_STABILITY': 'crimson'}, ax=ax)
        for _, row in self.df_feat_summary.head(8).iterrows():
            ax.annotate(row['feature_name'], (row['mean_absolute_shap'] + 0.3, row['mean_sign_stability']), fontsize=9)
        ax.set_title("Feature Attribution Stability vs. Attribution Magnitude")
        ax.set_xlabel("Mean Absolute SHAP Magnitude (µg/m³)")
        ax.set_ylabel("Attribution Sign Stability")
        plt.tight_layout()
        fig.savefig(plot_dir / "12_stability_vs_magnitude.png", dpi=150)
        plt.close(fig)

        logger.info("All 12 publication figures generated cleanly.")
