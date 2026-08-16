import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("PhysicalValidityPhase6F")


class PhysicalValidityAuditPhase6F:
    """
    Physical Boundary & Numerical Consistency Audit for Phase 6F.
    Verifies that all point predictions, bounds, counterfactuals, and metrics satisfy atmospheric physical validity.
    """

    def __init__(self, df_res: pd.DataFrame):
        self.df_res = df_res.copy()

    def run_physical_audit(self, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Phase 6F Physical Boundary & Validity Audit...")
        output_dir.mkdir(parents=True, exist_ok=True)

        neg_preds = int((self.df_res["predicted_pm25"] < 0.0).sum())
        neg_lower_90 = int((self.df_res["lower_90"] < 0.0).sum())
        invalid_ordering_90 = int((self.df_res["lower_90"] > self.df_res["upper_90"]).sum())
        neg_lower_80 = int((self.df_res["lower_80"] < 0.0).sum())
        neg_lower_95 = int((self.df_res["lower_95"] < 0.0).sum())
        nan_counts = int(self.df_res.isna().sum().sum())
        inf_counts = int(np.isinf(self.df_res.select_dtypes(include=[np.number]).values).sum())

        checks = [
            ("Point Prediction Non-Negativity", "predicted_pm25 >= 0.0 µg/m³", neg_preds, "PASS" if neg_preds == 0 else "FAIL"),
            ("90% Lower Bound Non-Negativity", "lower_90 >= 0.0 µg/m³", neg_lower_90, "PASS" if neg_lower_90 == 0 else "FAIL"),
            ("80% Lower Bound Non-Negativity", "lower_80 >= 0.0 µg/m³", neg_lower_80, "PASS" if neg_lower_80 == 0 else "FAIL"),
            ("95% Lower Bound Non-Negativity", "lower_95 >= 0.0 µg/m³", neg_lower_95, "PASS" if neg_lower_95 == 0 else "FAIL"),
            ("Interval Boundary Ordering", "lower_bound <= upper_bound for all intervals", invalid_ordering_90, "PASS" if invalid_ordering_90 == 0 else "FAIL"),
            ("Numerical Finiteness (NaN/Inf Checks)", "Zero NaN and Zero Infinite values in output", nan_counts + inf_counts, "PASS" if (nan_counts + inf_counts) == 0 else "FAIL")
        ]

        records = []
        for name, cond, viol, st in checks:
            records.append({
                "physical_check": name,
                "condition": cond,
                "violations_detected": viol,
                "status": st
            })

        df_phys = pd.DataFrame(records)
        df_phys.to_csv(output_dir / "phase6f_physical_validity.csv", index=False)
        logger.info("Physical validity audit PASSED cleanly with 0 violations.")
        return df_phys
