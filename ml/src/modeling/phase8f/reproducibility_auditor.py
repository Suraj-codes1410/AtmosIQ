"""
AtmosIQ Phase 8F: Numerical Reproducibility Auditor.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class Phase8FReproducibilityAuditor:
    """Audits numerical reproducibility and determinism across repeated evaluation runs."""

    def audit_reproducibility(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        if len(df1) != len(df2):
            df_fail = pd.DataFrame([{
                "column": "Row Count Match",
                "max_delta": float(abs(len(df1) - len(df2))),
                "status": "FAIL",
                "reason": "Unequal row count between execution runs"
            }])
            return False, df_fail

        num_cols = df1.select_dtypes(include=[np.number]).columns
        records = []
        max_overall_delta = 0.0

        for col in num_cols:
            if col in df2.columns:
                delta = float(np.max(np.abs(df1[col].values - df2[col].values)))
                max_overall_delta = max(max_overall_delta, delta)
                records.append({
                    "column": col,
                    "max_delta": delta,
                    "status": "PASS" if delta <= 1e-9 else "FAIL",
                    "reason": "Exact numerical reproducibility within tolerance <= 1e-9" if delta <= 1e-9 else "Numerical drift detected"
                })

        df_aud = pd.DataFrame(records)
        all_passed = bool(max_overall_delta <= 1e-9)
        return all_passed, df_aud
