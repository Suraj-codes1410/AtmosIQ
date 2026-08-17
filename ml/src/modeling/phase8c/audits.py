"""
AtmosIQ Phase 8C: Data Isolation, Integrity, and Reproducibility Audits.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IntegrityAndIsolationAuditor:
    """Executes formal integrity, isolation, and reproducibility audits."""

    def __init__(self, feature_registry: List[str], locked_eval_start_date: str = "2022-01-01"):
        self.feature_registry = list(feature_registry)
        self.locked_eval_start_date = locked_eval_start_date

    def audit_data_isolation(self, df_corpus: pd.DataFrame) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """Audits temporal isolation ensuring zero 2022–2024 records enter synthetic generation."""
        date_cols = [c for c in ["date", "synthetic_date", "timestamp"] if c in df_corpus.columns]
        violations = 0
        checks = []

        # Check 1: Real evaluation fold date leakage
        date_leaks = 0
        for col in ["date", "timestamp"]:
            if col in df_corpus.columns:
                real_dates = df_corpus[col].dropna().astype(str)
                leaked = df_corpus[real_dates.str.startswith(("2022", "2023", "2024"))]
                cnt = len(leaked)
                if cnt > 0:
                    date_leaks += cnt
                    violations += cnt
        
        if date_leaks > 0:
            checks.append({"audit_check": "Timestamp Leakage", "status": "FAIL", "violations": date_leaks, "details": f"Found evaluation fold dates >= {self.locked_eval_start_date}"})
        else:
            checks.append({"audit_check": "Timestamp Leakage", "status": "PASS", "violations": 0, "details": "Zero evaluation fold dates found"})

        # Check 2: Origin identifier
        synthetic_col = df_corpus.get("data_origin", None)
        if synthetic_col is not None:
            non_synth = int((synthetic_col != "synthetic").sum())
            if non_synth > 0: violations += non_synth
            checks.append({"audit_check": "Synthetic Origin Tag", "status": "PASS" if non_synth == 0 else "FAIL", "violations": non_synth, "details": "All observations tagged as synthetic"})

        # Check 3: Source partition label
        src_part = df_corpus.get("source_partition", None)
        if src_part is not None:
            unauthorized = int((src_part != "2020-2021").sum())
            if unauthorized > 0: violations += unauthorized
            checks.append({"audit_check": "Source Partition Tag", "status": "PASS" if unauthorized == 0 else "FAIL", "violations": unauthorized, "details": "Strictly restricted to 2020-2021 historical partition"})

        df_audit = pd.DataFrame(checks)
        passed = bool(violations == 0)

        summary = {
            "isolation_status": "PASS" if passed else "FAIL_LEAKAGE_DETECTED",
            "total_isolation_violations": violations,
            "locked_eval_start_date": self.locked_eval_start_date,
        }

        return passed, df_audit, summary

    def audit_corpus_integrity(self, df_corpus: pd.DataFrame) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """Audits schema, nulls, physical laws, hydrodynamic identity, and duplicate properties."""
        checks = []

        # 1. Schema completeness
        missing_feats = [f for f in self.feature_registry if f not in df_corpus.columns]
        checks.append({
            "dimension": "Feature Schema",
            "check": "35 Prediction-Safe Features Present",
            "violations": len(missing_feats),
            "status": "PASS" if len(missing_feats) == 0 else "FAIL",
            "details": "All 35 production features present" if len(missing_feats) == 0 else f"Missing: {missing_feats}"
        })

        # 2. NaN / Infinite Values
        nan_count = int(df_corpus.isna().sum().sum())
        inf_count = int(np.isinf(df_corpus.select_dtypes(include=[np.number]).values).sum())
        checks.append({
            "dimension": "Data Cleanliness",
            "check": "Zero NaN Values",
            "violations": nan_count,
            "status": "PASS" if nan_count == 0 else "FAIL",
            "details": f"{nan_count} NaN values detected"
        })
        checks.append({
            "dimension": "Data Cleanliness",
            "check": "Zero Infinite Values",
            "violations": inf_count,
            "status": "PASS" if inf_count == 0 else "FAIL",
            "details": f"{inf_count} infinite values detected"
        })

        # 3. Non-Negativity Laws
        neg_pm = int((df_corpus["pm25"] < 0.0).sum())
        neg_ws = int((df_corpus["wind_speed_kmh"] < 0.0).sum())
        neg_rain = int((df_corpus["rainfall_1d"] < 0.0).sum())
        bad_pblh = int((df_corpus["pblh_1d"] < 150.0).sum())
        checks.append({
            "dimension": "Physical Law",
            "check": "PM2.5 Non-Negativity (>= 0.0)",
            "violations": neg_pm,
            "status": "PASS" if neg_pm == 0 else "FAIL",
            "details": f"{neg_pm} negative PM2.5 rows"
        })
        checks.append({
            "dimension": "Physical Law",
            "check": "Meteorological Bounds (ws >= 0, rain >= 0, PBLH >= 150)",
            "violations": neg_ws + neg_rain + bad_pblh,
            "status": "PASS" if (neg_ws + neg_rain + bad_pblh) == 0 else "FAIL",
            "details": f"{neg_ws} bad ws, {neg_rain} bad rain, {bad_pblh} bad PBLH"
        })

        # 4. Hydrodynamic Identity (VI = ws_ms * PBLH)
        ws_ms = df_corpus["wind_speed_kmh"] * (1000.0 / 3600.0)
        expected_vi = ws_ms * df_corpus["pblh_1d"]
        vi_diff = np.abs(df_corpus["ventilation_index_1d"] - expected_vi)
        bad_vi = int((vi_diff > 1.0).sum())
        checks.append({
            "dimension": "Hydrodynamics",
            "check": "Ventilation Index Identity (VI = ws * PBLH)",
            "violations": bad_vi,
            "status": "PASS" if bad_vi == 0 else "FAIL",
            "details": f"{bad_vi} rows violated hydrodynamic identity"
        })

        # 5. Rain Event Binary Indicator Logic
        bad_rain = int(((df_corpus["rain_event_1d"] == 1) != (df_corpus["rainfall_1d"] >= 1.0)).sum())
        checks.append({
            "dimension": "Indicator Logic",
            "check": "Rain Event Indicator (rain >= 1.0mm)",
            "violations": bad_rain,
            "status": "PASS" if bad_rain == 0 else "FAIL",
            "details": f"{bad_rain} rows violated rain binary logic"
        })

        # 6. Trajectory Horizon Compliance (14 or 30 days only)
        traj_lens = df_corpus.groupby("trajectory_id").size()
        invalid_lens = int((~traj_lens.isin([14, 30])).sum())
        checks.append({
            "dimension": "Horizon Restriction",
            "check": "Trajectory Length in [14, 30] days",
            "violations": invalid_lens,
            "status": "PASS" if invalid_lens == 0 else "FAIL",
            "details": f"{invalid_lens} trajectories violated horizon restrictions"
        })

        df_audit = pd.DataFrame(checks)
        all_passed = bool((df_audit["status"] == "PASS").all())

        summary = {
            "integrity_status": "PASS" if all_passed else "FAIL",
            "total_checks": len(checks),
            "passed_checks": int((df_audit["status"] == "PASS").sum()),
            "failed_checks": int((df_audit["status"] == "FAIL").sum()),
        }

        return all_passed, df_audit, summary

    def audit_reproducibility(
        self,
        df_run1: pd.DataFrame,
        df_run2: pd.DataFrame
    ) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """Audits numerical delta across repeated release consolidation."""
        if len(df_run1) != len(df_run2):
            return False, pd.DataFrame([{"error": "Row count mismatch"}]), {"status": "FAIL_ROW_COUNT_MISMATCH"}

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

        summary = {
            "reproducibility_status": "PASS" if is_repro else "FAIL",
            "maximum_numerical_delta": max_delta,
            "tolerance": 1e-9,
        }

        return is_repro, df_audit, summary
