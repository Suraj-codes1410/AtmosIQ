import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationPhase4I")


class VisualizationEnginePhase4I:
    """
    Publication Visualizations Generator for Phase 4I.
    Generates 10 publication-quality figures in ml/experiments/phase4i/plots/.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    def generate_all_plots(
        self,
        merged_feat: pd.DataFrame,
        merged_grp: pd.DataFrame,
        v3_feat_df: pd.DataFrame,
        v3_grp_df: pd.DataFrame,
        ext_val_df: pd.DataFrame,
        seasonal_df: pd.DataFrame,
        stab_df: pd.DataFrame,
        cf_summary_df: pd.DataFrame,
        extreme_df: pd.DataFrame,
        consistency_df: pd.DataFrame
    ):
        logger.info("Generating Phase 4I Publication Plots...")

        # Plot 1: V2 vs V3 Feature Importance Comparison
        plt.figure(figsize=(10, 6))
        top15_feat = merged_feat.sort_values('v3_mean_abs_shap', ascending=False).head(15)
        df_p1 = top15_feat.melt(id_vars=['feature'], value_vars=['v2_mean_abs_shap', 'v3_mean_abs_shap'],
                                 var_name='Model_Version', value_name='Mean_Abs_SHAP')
        df_p1['Model_Version'] = df_p1['Model_Version'].replace({'v2_mean_abs_shap': 'v2 Production', 'v3_mean_abs_shap': 'v3 Promoted'})
        sns.barplot(data=df_p1, x='Mean_Abs_SHAP', y='feature', hue='Model_Version', palette='Blues_r')
        plt.title("Plot 1: V2 vs V3 Feature Importance (Top 15 Features)", fontsize=12, fontweight='bold')
        plt.xlabel("Mean |SHAP| (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "1_v2_vs_v3_feature_importance.png", dpi=300)
        plt.close()

        # Plot 2: V2 vs V3 Group Attribution Comparison
        plt.figure(figsize=(8, 5))
        df_p2 = merged_grp.melt(id_vars=['attribution_group'], value_vars=['v2_mean_abs_shap', 'v3_mean_abs_shap'],
                                 var_name='Model_Version', value_name='Mean_Abs_SHAP')
        df_p2['Model_Version'] = df_p2['Model_Version'].replace({'v2_mean_abs_shap': 'v2 Production', 'v3_mean_abs_shap': 'v3 Promoted'})
        sns.barplot(data=df_p2, x='Mean_Abs_SHAP', y='attribution_group', hue='Model_Version', palette='viridis')
        plt.title("Plot 2: V2 vs V3 Group Attribution Importance", fontsize=12, fontweight='bold')
        plt.xlabel("Mean |SHAP| (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "2_v2_vs_v3_group_attribution.png", dpi=300)
        plt.close()

        # Plot 3: V3 SHAP Summary (Top 20 Features)
        plt.figure(figsize=(10, 6))
        top20_v3 = v3_feat_df.head(20)
        sns.barplot(data=top20_v3, x='mean_abs_shap', y='feature', hue='group', dodge=False, palette='crest')
        plt.title("Plot 3: V3 TreeSHAP Global Feature Importance (Top 20)", fontsize=12, fontweight='bold')
        plt.xlabel("Mean |SHAP| (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "3_v3_shap_summary.png", dpi=300)
        plt.close()

        # Plot 4: V3 Group Attribution Breakdown
        plt.figure(figsize=(8, 5))
        sns.barplot(data=v3_grp_df, x='mean_abs_shap', y='attribution_group', palette='mako')
        plt.title("Plot 4: V3 Environmental Group SHAP Importance", fontsize=12, fontweight='bold')
        plt.xlabel("Mean |SHAP| (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "4_v3_group_attribution.png", dpi=300)
        plt.close()

        # Plot 5: External Feature Attribution
        plt.figure(figsize=(9, 5))
        sns.barplot(data=ext_val_df, x='mean_abs_shap', y='feature', palette='Purples_r')
        plt.title("Plot 5: External Environmental Features SHAP Magnitude", fontsize=12, fontweight='bold')
        plt.xlabel("Mean |SHAP| (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "5_external_feature_attribution.png", dpi=300)
        plt.close()

        # Plot 6: Seasonal Attribution Breakdown
        plt.figure(figsize=(9, 5))
        sns.barplot(data=seasonal_df, x='mean_mae_ugm3', y='season', palette='YlOrRd')
        plt.title("Plot 6: V3 Out-of-Sample MAE across Seasons", fontsize=12, fontweight='bold')
        plt.xlabel("MAE (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "6_seasonal_attribution.png", dpi=300)
        plt.close()

        # Plot 7: Temporal Multi-Year Stability
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=stab_df, x='year', y='mean_year_to_year_spearman_rho', marker='o', color='teal', linewidth=2.5)
        plt.title("Plot 7: Multi-Year Attribution Rank Stability (2020-2024)", fontsize=12, fontweight='bold')
        plt.ylabel("Spearman Rank Correlation (Rho)")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir / "7_temporal_stability.png", dpi=300)
        plt.close()

        # Plot 8: Counterfactual Scenario Comparison
        plt.figure(figsize=(10, 5))
        sns.barplot(data=cf_summary_df, x='mean_delta_pm25', y='scenario', palette='vlag')
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.title("Plot 8: V3 Counterfactual Mean Prediction Impact (ΔPM2.5)", fontsize=12, fontweight='bold')
        plt.xlabel("Mean ΔPM2.5 (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "8_counterfactual_comparison.png", dpi=300)
        plt.close()

        # Plot 9: Extreme Event Attribution
        plt.figure(figsize=(8, 4))
        sns.barplot(data=extreme_df, x='mae_ugm3', y='subset', palette='Reds_r')
        plt.title("Plot 9: V3 Model MAE on Extreme Pollution Subsets", fontsize=12, fontweight='bold')
        plt.xlabel("MAE (µg/m³)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "9_extreme_event_attribution.png", dpi=300)
        plt.close()

        # Plot 10: SHAP vs Counterfactual Consistency Rate
        plt.figure(figsize=(8, 4))
        sns.barplot(data=consistency_df, x='directional_consistency_rate', y='group', palette='crest')
        plt.axvline(0.944, color='red', linestyle='--', label='v2 Historical Baseline (94.4%)')
        plt.title("Plot 10: SHAP vs Counterfactual Directional Consistency Rate", fontsize=12, fontweight='bold')
        plt.xlabel("Directional Consistency Rate")
        plt.xlim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "10_shap_counterfactual_consistency.png", dpi=300)
        plt.close()

        logger.info(f"All 10 publication-quality plots generated in {self.output_dir}.")
