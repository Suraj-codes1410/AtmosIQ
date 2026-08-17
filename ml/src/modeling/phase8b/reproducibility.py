"""
AtmosIQ Phase 8B: Scaling Reproducibility Auditor.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Phase8BReproducibilityAuditor:
    """Audits numerical reproducibility across independent scaling runs."""

    def __init__(self):
        pass

    def run_reproducibility_audit(
        self,
        df_run1: pd.DataFrame,
        df_run2: pd.DataFrame
    ) -> Tuple[bool, float, pd.DataFrame]:
        logger.info("Executing Phase 8B Deterministic Reproducibility Audit...")

        if len(df_run1) == 0 or len(df_run2) == 0 or len(df_run1) != len(df_run2):
            return False, 1.0, pd.DataFrame()

        num_cols = df_run1.select_dtypes(include=[np.number]).columns
        max_delta = 0.0
        records = []

        for col in num_cols:
            if col in df_run2.columns:
                v1 = df_run1[col].values
                v2 = df_run2[col].values
                d = float(np.max(np.abs(v1 - v2)))
                max_delta = max(max_delta, d)
                records.append({
                    "column_name": col,
                    "max_delta": d,
                    "status": "PASS" if d <= 1e-9 else "FAIL"
                })

        df_audit = pd.DataFrame(records)
        is_repro = (max_delta <= 1e-9)

        logger.info(f"Reproducibility Audit completed. Max delta: {max_delta:.2e}, Status: {'PASS' if is_repro else 'FAIL'}")
        return is_repro, max_delta, df_audit
