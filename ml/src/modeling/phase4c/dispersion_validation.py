import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from ml.src.utils.logger import setup_logger

logger = setup_logger("DispersionValidationPhase4C")


class DispersionValidationPhase4C:
    """
    AtmosIQ Phase 4C Wind & Ventilation Dispersion Validator.
    Validates wind_ventilation SHAP attributions against surface wind speed indicators and low-ventilation stagnation regimes.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4c"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def validate_dispersion_attribution(self, df: pd.DataFrame, group_shap_df: pd.DataFrame) -> dict:
        """Executes statistical validation of wind_ventilation SHAP attributions."""
        logger.info("Executing Wind / Ventilation Dispersion Attribution Validation...")

        wind_shap = group_shap_df["wind_ventilation_shap"].values
        wind_col = "wind_speed_kmh_lag_1d" if "wind_speed_kmh_lag_1d" in df.columns else "wind_speed_kmh_roll_mean_7d"
        wind_vals = df[wind_col].values

        # Spearman & Pearson correlation (expect negative correlation: lower wind -> higher positive SHAP pollution contribution)
        spearman_corr, spearman_p = spearmanr(wind_shap, wind_vals)
        pearson_corr, pearson_p = pearsonr(wind_shap, wind_vals)

        logger.info(f"Wind SHAP vs {wind_col} -> Spearman: {spearman_corr:.4f} (p={spearman_p:.4e}), Pearson: {pearson_corr:.4f} (p={pearson_p:.4e}).")

        # Stagnation regime comparison (Low Wind <= 5 km/h vs High Wind >= 12 km/h)
        low_wind_mask = wind_vals <= 5.0
        high_wind_mask = wind_vals >= 12.0

        mean_shap_low_wind = float(np.mean(wind_shap[low_wind_mask])) if low_wind_mask.sum() > 0 else float(np.mean(wind_shap[wind_vals <= np.percentile(wind_vals, 25)]))
        mean_shap_high_wind = float(np.mean(wind_shap[high_wind_mask])) if high_wind_mask.sum() > 0 else float(np.mean(wind_shap[wind_vals >= np.percentile(wind_vals, 75)]))

        pos_freq_low_wind = float(np.mean(wind_shap[low_wind_mask] > 0)) if low_wind_mask.sum() > 0 else float(np.mean(wind_shap[wind_vals <= np.percentile(wind_vals, 25)] > 0))
        pos_freq_high_wind = float(np.mean(wind_shap[high_wind_mask] > 0)) if high_wind_mask.sum() > 0 else float(np.mean(wind_shap[wind_vals >= np.percentile(wind_vals, 75)] > 0))

        val_rows = [
            {"metric": "spearman_correlation", "value": float(spearman_corr), "p_value": float(spearman_p), "notes": "Expected negative: lower wind -> positive SHAP"},
            {"metric": "pearson_correlation", "value": float(pearson_corr), "p_value": float(pearson_p), "notes": "Linear correlation"},
            {"metric": "mean_shap_low_wind_stagnation", "value": mean_shap_low_wind, "p_value": None, "notes": "Low wind <= 5km/h regime"},
            {"metric": "mean_shap_high_wind_dispersion", "value": mean_shap_high_wind, "p_value": None, "notes": "High wind >= 12km/h regime"},
            {"metric": "pos_shap_freq_low_wind", "value": pos_freq_low_wind, "p_value": None, "notes": "Positive SHAP fraction low wind"},
            {"metric": "pos_shap_freq_high_wind", "value": pos_freq_high_wind, "p_value": None, "notes": "Positive SHAP fraction high wind"}
        ]

        val_df = pd.DataFrame(val_rows)
        val_df.to_csv(self.exp_dir / "wind_validation.csv", index=False)

        return {
            "spearman_corr": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "mean_shap_low_wind": mean_shap_low_wind,
            "mean_shap_high_wind": mean_shap_high_wind,
            "val_df": val_df
        }


if __name__ == "__main__":
    validator = DispersionValidationPhase4C()
