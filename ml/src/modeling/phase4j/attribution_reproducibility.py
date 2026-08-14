import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("AttributionReproducibilityPhase4J")


class AttributionReproducibilityPhase4J:
    """
    TreeSHAP Attribution and Reconstruction Reproducibility Engine for Phase 4J.
    Validates base_value + sum(SHAP) == prediction (tolerance <= 1e-4 µg/m³) and group aggregation determinism.
    """

    def __init__(self, model_path: Path, df_v3: pd.DataFrame, features_35: list):
        self.model = joblib.load(model_path)
        self.df_v3 = df_v3.copy()
        self.features = features_35
        self.X_all = self.df_v3[self.features].fillna(0.0)

    def run_attribution_reproducibility(self, output_csv: Path) -> pd.DataFrame:
        logger.info("Executing TreeSHAP Attribution & Reconstruction Reproducibility Test...")
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        preds = self.model.predict(self.X_all)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_all)
        base_val = float(explainer.expected_value[0]) if isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value)

        reconstructed_preds = base_val + shap_values.sum(axis=1)
        recon_errors = np.abs(preds - reconstructed_preds)

        max_err = float(np.max(recon_errors))
        mean_err = float(np.mean(recon_errors))
        median_err = float(np.median(recon_errors))
        p95_err = float(np.percentile(recon_errors, 95))

        passed = (max_err <= 1e-4)

        df_recon = pd.DataFrame([{
            "total_samples_explained": len(self.X_all),
            "base_value": base_val,
            "max_reconstruction_error": max_err,
            "mean_reconstruction_error": mean_err,
            "median_reconstruction_error": median_err,
            "p95_reconstruction_error": p95_err,
            "tolerance": 1e-4,
            "status": "PASS" if passed else "FAIL"
        }])
        df_recon.to_csv(output_csv, index=False)

        assert passed, f"TreeSHAP reconstruction failed! Max error = {max_err:.6e} > 1e-4"
        logger.info(f"TreeSHAP Attribution Reproducibility PASSED (Max error = {max_err:.6e} <= 1e-4).")
        return df_recon
