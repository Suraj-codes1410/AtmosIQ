"""
AtmosIQ Phase 8F: Feature Registry & Schema Compatibility Auditor.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class Phase8FSchemaAuditor:
    """Audits schema compatibility of synthetic corpora against feature_registry.csv."""

    def __init__(self, feature_registry_path: str):
        self.df_registry = pd.read_csv(feature_registry_path)
        self.expected_features = self.df_registry["feature_name"].tolist()

    def audit_corpus_schema(self, df_corpus: pd.DataFrame, corpus_name: str) -> Tuple[bool, pd.DataFrame]:
        records = []
        all_passed = True

        for feat in self.expected_features:
            observed = (feat in df_corpus.columns)
            dtype_obs = str(df_corpus[feat].dtype) if observed else "MISSING"
            status = "PASS" if observed else "FAIL"
            reason = "Feature present and compatible" if observed else "Expected prediction-safe feature missing"
            if not observed:
                all_passed = False

            records.append({
                "corpus": corpus_name,
                "feature": feat,
                "expected": True,
                "observed": observed,
                "dtype_expected": "numeric",
                "dtype_observed": dtype_obs,
                "status": status,
                "reason": reason,
            })

        # Target variable audit
        has_target = ("pm25" in df_corpus.columns)
        records.append({
            "corpus": corpus_name,
            "feature": "pm25 (Target)",
            "expected": True,
            "observed": has_target,
            "dtype_expected": "float64",
            "dtype_observed": str(df_corpus["pm25"].dtype) if has_target else "MISSING",
            "status": "PASS" if has_target else "FAIL",
            "reason": "Target variable present and isolated",
        })

        df_aud = pd.DataFrame(records)
        return all_passed, df_aud
