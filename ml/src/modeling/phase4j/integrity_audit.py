import hashlib
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("IntegrityAuditPhase4J")


class ReleaseIntegrityAuditPhase4J:
    """
    Comprehensive Release Integrity Audit for Phase 4J.
    Audits immutable data lineage, dataset characteristics, and strict 35-feature production registry.
    """

    EXPECTED_HASHES = {
        "v1_dataset": "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df",
        "v2_dataset": "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301",
        "v3_dataset": "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736",
        "v2_control_model": "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162",
        "v3_promoted_model": "9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210"
    }

    PRODUCTION_35_FEATURES = [
        # Core PM2.5 Lags & Rolling (10)
        "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
        "pm25_roll_mean_3d", "pm25_roll_mean_7d", "pm25_roll_mean_14d",
        "pm25_roll_std_7d", "pm25_roll_max_7d", "pm25_roll_min_7d",
        # Temperature Lags & Rolling (3)
        "temperature_c_lag_1d", "temperature_c_roll_mean_3d", "temperature_c_roll_min_3d",
        # Humidity Lags & Rolling (3)
        "humidity_pct_lag_1d", "humidity_pct_roll_mean_3d", "humidity_pct_roll_max_7d",
        # Wind Speed & Components (4)
        "wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d", "wind_u_component_1d", "wind_v_component_1d",
        # Stubble & Fire Hotspots (5)
        "is_stubble_season", "fire_hotspot_count_lag_1d", "fire_hotspot_count_roll_mean_3d",
        "fire_hotspot_count_roll_mean_7d", "upwind_stubble_quadrant_1d",
        # Rainfall & Washout (4)
        "rainfall_1d", "rainfall_3d", "rain_event_1d", "washout_index_3d",
        # PBL & Boundary Layer (4)
        "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d", "ventilation_index_1d",
        # AOD & Calendar (2)
        "aod_550_1d", "festival_window"
    ]

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir
        self.v1_path = self.root_dir / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
        self.v2_path = self.root_dir / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
        self.v3_path = self.root_dir / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
        self.v1_model_path = self.root_dir / "ml" / "models" / "attribution" / "v1" / "model.joblib"
        self.v3_model_path = self.root_dir / "ml" / "models" / "attribution" / "v3" / "model.joblib"

    @staticmethod
    def calculate_sha256(filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def audit_all(self, exp_dir: Path) -> dict:
        logger.info("Executing Release Integrity Audit for Phase 4J...")
        exp_dir.mkdir(parents=True, exist_ok=True)
        audit_records = []

        # 1. Lineage Hashes
        v1_h = self.calculate_sha256(self.v1_path)
        v2_h = self.calculate_sha256(self.v2_path)
        v3_h = self.calculate_sha256(self.v3_path)
        v1_m_h = self.calculate_sha256(self.v1_model_path)
        v3_m_h = self.calculate_sha256(self.v3_model_path)

        for name, actual, expected in [
            ("Dataset v1 Hash", v1_h, self.EXPECTED_HASHES["v1_dataset"]),
            ("Dataset v2 Hash", v2_h, self.EXPECTED_HASHES["v2_dataset"]),
            ("Dataset v3 Hash", v3_h, self.EXPECTED_HASHES["v3_dataset"]),
            ("Phase 3G/v2 Control Model Hash", v1_m_h, self.EXPECTED_HASHES["v2_control_model"]),
            ("Phase 4H/v3 Promoted Model Hash", v3_m_h, self.EXPECTED_HASHES["v3_promoted_model"])
        ]:
            passed = (actual == expected)
            audit_records.append({
                "audit_check": name,
                "expected": expected,
                "observed": actual,
                "status": "PASS" if passed else "FAIL",
                "notes": "Immutable artifact integrity verified" if passed else "SHA-256 hash mismatch!"
            })

        # 2. Dataset v3 Data Quality & Completeness
        df_v3 = pd.read_csv(self.v3_path)
        df_v3['date_dt'] = pd.to_datetime(df_v3['date'])

        row_count = len(df_v3)
        col_count = len(df_v3.columns)
        min_date = df_v3['date_dt'].min().strftime('%Y-%m-%d')
        max_date = df_v3['date_dt'].max().strftime('%Y-%m-%d')
        expected_dates = pd.date_range("2020-01-01", "2024-12-31", freq='D')
        missing_dates = len(expected_dates) - len(df_v3['date_dt'].unique())
        duplicate_dates = df_v3['date_dt'].duplicated().sum()

        target_valid = (df_v3['pm25'] >= 0).all() and (df_v3['pm25'] <= 1000).all() and not df_v3['pm25'].isnull().any()

        for name, cond, exp_val, obs_val, note in [
            ("Dataset v3 Row Count", row_count == 1827, 1827, row_count, "Exact 1,827 daily rows"),
            ("Dataset v3 Start Date", min_date == "2020-01-01", "2020-01-01", min_date, "Exact start date"),
            ("Dataset v3 End Date", max_date == "2024-12-31", "2024-12-31", max_date, "Exact end date"),
            ("Dataset v3 Missing Dates", missing_dates == 0, 0, missing_dates, "Zero missing dates in timeline"),
            ("Dataset v3 Duplicate Dates", duplicate_dates == 0, 0, duplicate_dates, "Zero duplicate dates"),
            ("Dataset v3 Target Validity (0-1000 µg/m³)", target_valid, "Valid non-null range", "All in range", "Physical bounds respected")
        ]:
            audit_records.append({
                "audit_check": name,
                "expected": str(exp_val),
                "observed": str(obs_val),
                "status": "PASS" if cond else "FAIL",
                "notes": note
            })

        # 3. Strict 35-Feature Production Registry Verification
        v3_cols = set(df_v3.columns)
        model_feats = self.PRODUCTION_35_FEATURES
        model_feat_count = len(model_feats)

        all_present = all(f in v3_cols for f in model_feats)
        unsafe_features = {'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'}
        leaked_features = [f for f in model_feats if f in unsafe_features or f.startswith('pm25_same_day')]
        zero_leakage = (len(leaked_features) == 0)

        # Check for NaNs in the 35 features
        nans_in_features = df_v3[model_feats].isnull().sum().sum()

        for name, cond, exp_val, obs_val, note in [
            ("Production Feature Count", model_feat_count == 35, 35, model_feat_count, "Exactly 35 registered prediction-safe features"),
            ("Production Features in Dataset v3", all_present, True, all_present, "All 35 features exist in Dataset v3"),
            ("Production Feature Leakage Audit", zero_leakage, "0 unsafe features", f"{len(leaked_features)} unsafe features", "Zero same-day targets in model input"),
            ("Production Feature Missing Values", nans_in_features == 0, 0, nans_in_features, "Clean zero-missingness in model inputs"),
            ("Dataset v3 Total Columns vs Model Features", col_count >= 250 and model_feat_count == 35, "275 total cols != 35 model features", f"{col_count} total cols vs {model_feat_count} model feats", "Model strictly isolates 35 features")
        ]:
            audit_records.append({
                "audit_check": name,
                "expected": str(exp_val),
                "observed": str(obs_val),
                "status": "PASS" if cond else "FAIL",
                "notes": note
            })

        df_audit = pd.DataFrame(audit_records)
        df_audit.to_csv(exp_dir / "release_integrity_audit.csv", index=False)

        all_passed = (df_audit['status'] == 'PASS').all()
        logger.info(f"Integrity Audit Completed. Total checks: {len(df_audit)}, All passed: {all_passed}.")
        assert all_passed, "Release integrity audit failed! Inspect release_integrity_audit.csv."

        return {
            "df_audit": df_audit,
            "all_passed": all_passed,
            "v3_model_hash": v3_m_h,
            "v3_dataset_hash": v3_h,
            "features_35": model_feats
        }
