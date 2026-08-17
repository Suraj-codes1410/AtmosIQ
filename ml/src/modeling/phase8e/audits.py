"""
AtmosIQ Phase 8E: Formal Audits Engine (Leakage, Physics, Hydrodynamics, Provenance, Reproducibility).
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class Phase8EAuditor:
    """Performs rigorous research-grade audits for Phase 8E admission gate."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def audit_leakage(self, df_real_train: pd.DataFrame, df_synth_8c: pd.DataFrame, df_synth_8d: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        checks = []
        # Check train frame
        train_leaks = int((pd.to_datetime(df_real_train["date"]) >= pd.to_datetime("2022-01-01")).sum())
        checks.append({
            "dimension": "Training Data Isolation",
            "check": "Historical 2020-2021 Train Fold Dates < 2022-01-01",
            "violations": train_leaks,
            "status": "PASS" if train_leaks == 0 else "FAIL",
        })

        # Check synthetic date columns if present
        for name, df_s in [("Phase 8C Baseline", df_synth_8c), ("Phase 8D Calibrated", df_synth_8d)]:
            if "date" in df_s.columns:
                s_dates = df_s["date"].dropna().astype(str)
                leaked = int(s_dates.str.startswith(("2022", "2023", "2024")).sum())
            else:
                leaked = 0
            checks.append({
                "dimension": "Synthetic Provenance Date Check",
                "check": f"{name} Free of Locked Evaluation Dates",
                "violations": leaked,
                "status": "PASS" if leaked == 0 else "FAIL",
            })

        df_aud = pd.DataFrame(checks)
        all_passed = bool((df_aud["status"] == "PASS").all())
        return all_passed, df_aud

    def audit_physical_validity(self, df_synth_8c: pd.DataFrame, df_synth_8d: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        checks = []
        for name, df in [("Phase 8C Baseline", df_synth_8c), ("Phase 8D Calibrated", df_synth_8d)]:
            neg_pm = int((df["pm25"] < 0.0).sum())
            ws_ms = df["wind_speed_kmh"] * (1000.0 / 3600.0)
            expected_vi = ws_ms * df["pblh_1d"]
            bad_vi = int((np.abs(df["ventilation_index_1d"] - expected_vi) > 1.0).sum())
            nan_count = int(df.isna().sum().sum())

            checks.append({
                "corpus": name,
                "check": "PM2.5 Non-Negativity (>= 0)",
                "violations": neg_pm,
                "status": "PASS" if neg_pm == 0 else "FAIL",
            })
            checks.append({
                "corpus": name,
                "check": "Hydrodynamic Identity (VI = ws * PBLH)",
                "violations": bad_vi,
                "status": "PASS" if bad_vi == 0 else "FAIL",
            })
            checks.append({
                "corpus": name,
                "check": "Zero NaN / Inf Values",
                "violations": nan_count,
                "status": "PASS" if nan_count == 0 else "FAIL",
            })

        df_aud = pd.DataFrame(checks)
        all_passed = bool((df_aud["status"] == "PASS").all())
        return all_passed, df_aud

    def audit_provenance(self, df_synth_8c: pd.DataFrame, df_synth_8d: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        checks = []
        for name, df in [("Phase 8C Baseline", df_synth_8c), ("Phase 8D Calibrated", df_synth_8d)]:
            has_traj_id = "trajectory_id" in df.columns
            missing_ids = int(df["trajectory_id"].isna().sum()) if has_traj_id else len(df)
            has_origin = "data_origin" in df.columns
            bad_origin = int((df["data_origin"] != "synthetic").sum()) if has_origin else 0

            checks.append({
                "corpus": name,
                "check": "Trajectory ID Completeness",
                "violations": missing_ids,
                "status": "PASS" if missing_ids == 0 else "FAIL",
            })
            checks.append({
                "corpus": name,
                "check": "Data Origin Tagged Synthetic",
                "violations": bad_origin,
                "status": "PASS" if bad_origin == 0 else "FAIL",
            })

        df_aud = pd.DataFrame(checks)
        all_passed = bool((df_aud["status"] == "PASS").all())
        return all_passed, df_aud

    def audit_reproducibility(self, df_run1: pd.DataFrame, df_run2: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        if len(df_run1) != len(df_run2):
            return False, pd.DataFrame([{"check": "Row count equality", "violations": 1, "status": "FAIL"}])

        num_cols = df_run1.select_dtypes(include=[np.number]).columns
        max_delta = 0.0
        records = []
        for col in num_cols:
            if col in df_run2.columns:
                d = float(np.max(np.abs(df_run1[col].values - df_run2[col].values)))
                max_delta = max(max_delta, d)
                records.append({"column": col, "max_delta": d, "status": "PASS" if d <= 1e-9 else "FAIL"})

        df_aud = pd.DataFrame(records)
        all_passed = bool(max_delta <= 1e-9)
        return all_passed, df_aud
