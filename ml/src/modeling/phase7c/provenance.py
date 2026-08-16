"""
AtmosIQ Phase 7C: Provenance and Isolation Auditor.
"""

import pandas as pd
from typing import Dict, Any


class ProvenanceAuditorPhase7C:
    def __init__(self, dev_train_end: str, locked_eval_start: str):
        self.dev_train_end = dev_train_end
        self.locked_eval_start = locked_eval_start

    def audit_provenance(self, df_synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Audits synthetic provenance fields and metadata completeness."""
        required_cols = [
            "data_origin", "generator_version", "trajectory_id",
            "synthetic_timestamp", "random_seed", "generation_timestamp"
        ]
        has_all_cols = all(c in df_synthetic.columns for c in required_cols)
        origin_correct = (df_synthetic["data_origin"] == "synthetic").all()
        version_correct = (df_synthetic["generator_version"].str.contains("HP-STG")).all()
        zero_nans = df_synthetic[required_cols].isna().sum().sum() == 0

        passed = has_all_cols and origin_correct and version_correct and zero_nans

        return {
            "provenance_passed": passed,
            "has_all_required_columns": has_all_cols,
            "all_data_origin_synthetic": origin_correct,
            "generator_version_verified": version_correct,
            "total_synthetic_records": len(df_synthetic),
            "num_trajectories": df_synthetic["trajectory_id"].nunique(),
        }
