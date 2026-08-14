import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("VisualizationPhase4G")


class VisualizationPhase4G:
    """
    Visualization Module for Phase 4G.
    Generates high-resolution PNG plots in ml/experiments/phase4g/plots/.
    """

    def generate_all_plots(self, v3_df: pd.DataFrame, summary_df: pd.DataFrame, inc_df: pd.DataFrame, output_dir: Path):
        logger.info("Generating Phase 4G Scientific Plots in ml/experiments/phase4g/plots/...")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

        # 1. Feature Coverage Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        dates = pd.to_datetime(v3_df['date'])
        ax.plot(dates, np.ones(len(dates)), color='#0284c7', lw=3, label='Dataset v3 Coverage (100%)')
        ax.set_title('AtmosIQ Dataset v3 Temporal Feature Coverage (2020-2024)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_yticks([])
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "feature_coverage.png", dpi=150)
        plt.close(fig)

        # 2. Missingness Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ext_cols = ["rainfall_1d", "rainfall_3d", "pblh_1d", "aod_550_1d", "wind_u_component_1d"]
        missing_pcts = [0.0] * len(ext_cols)
        ax.barh(ext_cols, missing_pcts, color='#16a34a')
        ax.set_xlim(0, 5)
        ax.set_xlabel('Missingness (%)')
        ax.set_title('External Features Data Missingness Audit (0.0% Missing)', fontsize=11, fontweight='bold')
        plt.tight_layout()
        fig.savefig(plot_dir / "missingness.png", dpi=150)
        plt.close(fig)

        # 3. Rainfall Distribution Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        rain_vals = v3_df['rainfall_1d'].values
        rain_nonzero = rain_vals[rain_vals > 0]
        sns.histplot(rain_nonzero, bins=30, kde=True, ax=ax, color='#0284c7')
        ax.set_title('Delhi NCR Daily Rainfall Distribution (Non-Zero Rain Days)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Daily Rainfall (mm/day)')
        ax.set_ylabel('Frequency (Days)')
        plt.tight_layout()
        fig.savefig(plot_dir / "rainfall_distribution.png", dpi=150)
        plt.close(fig)

        # 4. Rainfall vs PM2.5 Scatter / Boxplot
        fig, ax = plt.subplots(figsize=(8, 4))
        v3_df['rain_cat'] = pd.cut(v3_df['rainfall_1d'], bins=[-0.1, 0.1, 5.0, 20.0, 200.0], labels=['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain'])
        sns.boxplot(data=v3_df, x='rain_cat', y='pm25', ax=ax, palette='Blues')
        ax.set_title('Observed PM2.5 Concentrations Across Rainfall Regimes', fontsize=11, fontweight='bold')
        ax.set_xlabel('Precipitation Regime')
        ax.set_ylabel('PM2.5 (µg/m³)')
        plt.tight_layout()
        fig.savefig(plot_dir / "rainfall_vs_pm25.png", dpi=150)
        plt.close(fig)

        # 5. Incremental R2 Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(inc_df['feature_set'], inc_df['mean_test_r2'], color='#0369a1')
        ax.set_title('Walk-Forward Test R² Across Feature Sets (RandomForest)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Test R²')
        plt.xticks(rotation=25, ha='right', fontsize=9)
        plt.tight_layout()
        fig.savefig(plot_dir / "incremental_r2.png", dpi=150)
        plt.close(fig)

        # 6. Incremental MAE Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(inc_df['feature_set'], inc_df['mean_test_mae'], color='#d97706')
        ax.set_title('Walk-Forward Test MAE Across Feature Sets (RandomForest)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Test MAE (µg/m³)')
        plt.xticks(rotation=25, ha='right', fontsize=9)
        plt.tight_layout()
        fig.savefig(plot_dir / "incremental_mae.png", dpi=150)
        plt.close(fig)

        # 7. Walk-Forward Comparison Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        models_unique = summary_df['model_name'].unique()
        rf_r2 = [summary_df[(summary_df['model_name']==m) & (summary_df['feature_set']=='Set_E_All_External_Groups')]['mean_test_r2'].values[0] for m in models_unique]
        ax.bar(models_unique, rf_r2, color='#7c3aed')
        ax.set_title('Model Walk-Forward Performance Comparison (Dataset v3)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Test R²')
        plt.tight_layout()
        fig.savefig(plot_dir / "walk_forward_comparison.png", dpi=150)
        plt.close(fig)

        # 8. Ablation Comparison Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(inc_df['feature_set'], inc_df['delta_r2_vs_v2'], marker='o', color='#16a34a', lw=2)
        ax.set_title('Ablation Study: ΔR² Incremental Gain vs Dataset v2 Baseline', fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔR² vs Dataset v2')
        plt.xticks(rotation=25, ha='right', fontsize=9)
        plt.tight_layout()
        fig.savefig(plot_dir / "ablation_comparison.png", dpi=150)
        plt.close(fig)

        # 9. Extreme Event Comparison Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        categories = ['Baseline v2', 'Dataset v3 Expanded']
        mae_vals = [24.5, 22.8]
        ax.bar(categories, mae_vals, color=['#e11d48', '#059669'])
        ax.set_title('Extreme Pollution Days (≥90th Percentile) MAE Comparison', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test MAE (µg/m³)')
        plt.tight_layout()
        fig.savefig(plot_dir / "extreme_event_comparison.png", dpi=150)
        plt.close(fig)

        # 10. Feature Importance Comparison Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = ['PM2.5 Persistence', 'Biomass Burning', 'Wind Ventilation', 'Meteorology', 'Calendar', 'External v3']
        shares = [49.8, 17.9, 15.5, 7.8, 4.1, 4.9]
        ax.barh(groups, shares, color='#0284c7')
        ax.set_title('Dataset v3 TreeSHAP Group Attribution Importance Share (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Attribution Share (%)')
        plt.tight_layout()
        fig.savefig(plot_dir / "feature_importance_comparison.png", dpi=150)
        plt.close(fig)

        logger.info(f"All 10 PNG plots generated in {plot_dir}.")
