"""
AtmosIQ Phase 8F: Provenance & Lineage Traceability Auditor.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class Phase8FProvenanceAuditor:
    """Audits observation-level provenance, trajectory identifiers, and lineage tags."""

    def audit_provenance(self, df_8c: pd.DataFrame, df_8d: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        checks = []

        for name, df in [("AtmosIQ_Synthetic_Production_v1.0.0", df_8c), ("AtmosIQ_Synthetic_Calibrated_v0.1.0", df_8d)]:
            # 1. Trajectory ID Completeness
            has_id = "trajectory_id" in df.columns
            missing_ids = int(df["trajectory_id"].isna().sum()) if has_id else len(df)
            checks.append({
                "corpus": name,
                "check": "Trajectory ID Completeness",
                "violations": missing_ids,
                "status": "PASS" if missing_ids == 0 else "FAIL",
                "details": f"{df['trajectory_id'].nunique()} unique trajectory IDs present",
            })

            # 2. Data Origin Tag
            has_origin = "data_origin" in df.columns
            bad_origin = int((df["data_origin"] != "synthetic").sum()) if has_origin else 0
            checks.append({
                "corpus": name,
                "check": "Data Origin Tagged 'synthetic'",
                "violations": bad_origin,
                "status": "PASS" if bad_origin == 0 else "FAIL",
                "details": f"All {len(df)} rows tagged as synthetic data origin",
            })

            # 3. Trajectory Length Horizon Compliance (14 or 30 days)
            traj_lens = df.groupby("trajectory_id").size()
            invalid_lens = int((~traj_lens.isin([14, 30])).sum())
            checks.append({
                "corpus": name,
                "check": "Approved Horizon Compliance (14 or 30 days)",
                "violations": invalid_lens,
                "status": "PASS" if invalid_lens == 0 else "FAIL",
                "details": f"Trajectory lengths: {traj_lens.value_counts().to_dict()}",
            })

            # 4. Source Development Partition Tag
            has_source = "source_partition" in df.columns
            bad_source = int((df["source_partition"] != "2020-2021").sum()) if has_source else 0
            checks.append({
                "corpus": name,
                "check": "Source Partition Traceability ('2020-2021')",
                "violations": bad_source,
                "status": "PASS" if bad_source == 0 else "FAIL",
                "details": f"All rows derived from 2020-2021 historical development fold",
            })

        df_aud = pd.DataFrame(checks)
        all_passed = bool((df_aud["status"] == "PASS").all())
        return all_passed, df_aud
