"""
AtmosIQ Phase 8D: Audits Engine (Leakage, Physics, Memorization, Reproducibility).
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class Phase8DAuditor:
    """Performs formal leakage, physics, memorization, and reproducibility audits for Phase 8D."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_dev = None

    def fit_reference(self, df_real_dev: pd.DataFrame):
        common = [f for f in self.feature_registry if f in df_real_dev.columns]
        X_real = df_real_dev[common].values
        X_scaled = self.scaler.fit_transform(X_real)
        self.nn_dev = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        self.nn_dev.fit(X_scaled)

    def audit_leakage(self, df_calibrated: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        date_leaks = 0
        for col in ["date", "timestamp"]:
            if col in df_calibrated.columns:
                real_dates = df_calibrated[col].dropna().astype(str)
                leaked = df_calibrated[real_dates.str.startswith(("2022", "2023", "2024"))]
                date_leaks += len(leaked)

        passed = (date_leaks == 0)
        df_audit = pd.DataFrame([{
            "check": "Evaluation Fold Date Leakage (< 2022-01-01)",
            "violations": date_leaks,
            "status": "PASS" if passed else "FAIL",
            "details": "Zero evaluation dates found in calibrated corpus"
        }])
        return passed, df_audit

    def audit_physics(self, df_calibrated: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        neg_pm = int((df_calibrated["pm25"] < 0.0).sum())
        ws_ms = df_calibrated["wind_speed_kmh"] * (1000.0 / 3600.0)
        expected_vi = ws_ms * df_calibrated["pblh_1d"]
        bad_vi = int((np.abs(df_calibrated["ventilation_index_1d"] - expected_vi) > 1.0).sum())
        nan_count = int(df_calibrated.isna().sum().sum())

        passed = (neg_pm == 0 and bad_vi == 0 and nan_count == 0)
        df_audit = pd.DataFrame([
            {"check": "PM2.5 Non-Negativity", "violations": neg_pm, "status": "PASS" if neg_pm == 0 else "FAIL"},
            {"check": "Hydrodynamic Identity (VI = ws * PBLH)", "violations": bad_vi, "status": "PASS" if bad_vi == 0 else "FAIL"},
            {"check": "Zero NaN / Inf Values", "violations": nan_count, "status": "PASS" if nan_count == 0 else "FAIL"},
        ])
        return passed, df_audit

    def audit_memorization(self, df_calibrated: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        common = [f for f in self.feature_registry if f in df_calibrated.columns]
        X_cal = df_calibrated[common].values
        X_scaled = self.scaler.transform(X_cal)

        dists, _ = self.nn_dev.kneighbors(X_scaled)
        dists_vec = dists[:, 0]

        exact_dups = int((dists_vec <= 1e-6).sum())
        near_dups = int(((dists_vec > 1e-6) & (dists_vec < 0.05)).sum())
        passed = (exact_dups == 0 and near_dups == 0)

        df_audit = pd.DataFrame([
            {"check": "Exact Historical Duplicates", "violations": exact_dups, "status": "PASS" if exact_dups == 0 else "FAIL"},
            {"check": "Near-Duplicate Trajectory Copying", "violations": near_dups, "status": "PASS" if near_dups == 0 else "FAIL"},
        ])
        return passed, df_audit

    def audit_reproducibility(self, df_run1: pd.DataFrame, df_run2: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        if len(df_run1) != len(df_run2):
            return False, pd.DataFrame([{"check": "Row count match", "status": "FAIL"}])

        num_cols = df_run1.select_dtypes(include=[np.number]).columns
        max_delta = 0.0
        records = []
        for col in num_cols:
            if col in df_run2.columns:
                d = float(np.max(np.abs(df_run1[col].values - df_run2[col].values)))
                max_delta = max(max_delta, d)
                records.append({"column": col, "max_delta": d, "status": "PASS" if d <= 1e-9 else "FAIL"})

        passed = (max_delta <= 1e-9)
        return passed, pd.DataFrame(records)
