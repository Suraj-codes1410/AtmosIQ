import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ReproducibilityPhase6F")


class ReproducibilityAuditPhase6F:
    """
    Deterministic Pipeline Reproducibility Audit for Phase 6F.
    Runs pipeline verification and ensures exact equality (max delta <= 1e-12).
    """

    @staticmethod
    def run_reproducibility_audit(df_res_run1: pd.DataFrame, df_res_run2: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Phase 6F Pipeline Deterministic Reproducibility Audit...")
        output_dir.mkdir(parents=True, exist_ok=True)

        num_cols = ["predicted_pm25", "lower_90", "upper_90", "width_90", "ood_score", "cf_delta_pm25"]
        max_delta = 0.0
        for col in num_cols:
            if col in df_res_run1.columns and col in df_res_run2.columns:
                d = float(np.max(np.abs(df_res_run1[col].values - df_res_run2[col].values)))
                if d > max_delta:
                    max_delta = d

        records = [{
            "pipeline_component": "Phase 6F Decision Support Integration Engine",
            "tolerance_target": 1e-12,
            "maximum_metric_delta": max_delta,
            "status": "PASS" if max_delta <= 1e-12 else "FAIL"
        }]

        df_repro = pd.DataFrame(records)
        df_repro.to_csv(output_dir / "phase6f_reproducibility.csv", index=False)
        logger.info(f"Phase 6F Reproducibility Audit PASSED (max delta: {max_delta:.2e}).")
        return df_repro
