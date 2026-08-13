import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from ml.src.utils.logger import setup_logger

logger = setup_logger("BiomassValidationPhase4C")


class BiomassValidationPhase4C:
    """
    AtmosIQ Phase 4C Biomass Burning Attribution Validator.
    Validates biomass_burning SHAP values against satellite MODIS/VIIRS fire hotspot indicators using Spearman/Pearson correlations, quantile comparisons, and conditional agreement probabilities.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def validate_biomass_attribution(self, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> dict:
        """Executes statistical validation of biomass_burning SHAP attributions."""
        logger.info("Executing Biomass Burning Attribution Validation...")

        biomass_shap = group_shap_df["biomass_burning_shap"].values
        fire_col = "fire_hotspot_count_lag_1d" if "fire_hotspot_count_lag_1d" in df.columns else "fire_hotspot_count_roll_mean_7d"
        fire_vals = df[fire_col].values

        # A. Spearman & Pearson Correlation Analysis
        spearman_corr, spearman_p = spearmanr(biomass_shap, fire_vals)
        pearson_corr, pearson_p = pearsonr(biomass_shap, fire_vals)

        logger.info(f"Biomass SHAP vs {fire_col} -> Spearman: {spearman_corr:.4f} (p={spearman_p:.4e}), Pearson: {pearson_corr:.4f} (p={pearson_p:.4e}).")

        # B. Quantile Analysis (Low <=25th, Normal 25th-75th, High >=75th)
        q25, q75 = np.percentile(fire_vals, 25), np.percentile(fire_vals, 75)

        low_mask = fire_vals <= q25
        norm_mask = (fire_vals > q25) & (fire_vals < q75)
        high_mask = fire_vals >= q75

        mean_shap_low = float(np.mean(biomass_shap[low_mask]))
        mean_shap_norm = float(np.mean(biomass_shap[norm_mask]))
        mean_shap_high = float(np.mean(biomass_shap[high_mask]))

        med_shap_low = float(np.median(biomass_shap[low_mask]))
        med_shap_norm = float(np.median(biomass_shap[norm_mask]))
        med_shap_high = float(np.median(biomass_shap[high_mask]))

        pos_freq_low = float(np.mean(biomass_shap[low_mask] > 0))
        pos_freq_high = float(np.mean(biomass_shap[high_mask] > 0))

        # C. Conditional Agreement Probabilities
        high_shap_thresh = np.percentile(biomass_shap, 75)
        high_shap_mask = biomass_shap >= high_shap_thresh

        p_high_shap_given_high_fire = float(np.mean(high_shap_mask[high_mask]))
        p_high_fire_given_high_shap = float(np.mean(high_mask[high_shap_mask]))

        logger.info(f"P(High SHAP | High Fire): {p_high_shap_given_high_fire*100:.1f}%, P(High Fire | High SHAP): {p_high_fire_given_high_shap*100:.1f}%.")

        val_rows = [
            {"metric": "spearman_correlation", "value": float(spearman_corr), "p_value": float(spearman_p), "notes": "Primary robust correlation"},
            {"metric": "pearson_correlation", "value": float(pearson_corr), "p_value": float(pearson_p), "notes": "Linear correlation"},
            {"metric": "mean_shap_low_fire_q25", "value": mean_shap_low, "p_value": None, "notes": f"Low fire threshold <= {q25:.1f}"},
            {"metric": "mean_shap_normal_fire", "value": mean_shap_norm, "p_value": None, "notes": f"Normal fire range {q25:.1f} to {q75:.1f}"},
            {"metric": "mean_shap_high_fire_q75", "value": mean_shap_high, "p_value": None, "notes": f"High fire threshold >= {q75:.1f}"},
            {"metric": "median_shap_low_fire", "value": med_shap_low, "p_value": None, "notes": "Median SHAP low fire"},
            {"metric": "median_shap_high_fire", "value": med_shap_high, "p_value": None, "notes": "Median SHAP high fire"},
            {"metric": "pos_shap_freq_low_fire", "value": pos_freq_low, "p_value": None, "notes": "Positive SHAP fraction low fire"},
            {"metric": "pos_shap_freq_high_fire", "value": pos_freq_high, "p_value": None, "notes": "Positive SHAP fraction high fire"},
            {"metric": "p_high_shap_given_high_fire", "value": p_high_shap_given_high_fire, "p_value": None, "notes": "Conditional probability"},
            {"metric": "p_high_fire_given_high_shap", "value": p_high_fire_given_high_shap, "p_value": None, "notes": "Conditional probability"}
        ]

        val_df = pd.DataFrame(val_rows)
        val_df.to_csv(self.exp_dir / "biomass_validation.csv", index=False)

        return {
            "spearman_corr": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "mean_shap_low": mean_shap_low,
            "mean_shap_high": mean_shap_high,
            "p_high_shap_given_high_fire": p_high_shap_given_high_fire,
            "val_df": val_df
        }


if __name__ == "__main__":
    validator = BiomassValidationPhase4C()
