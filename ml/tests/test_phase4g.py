import hashlib
from pathlib import Path
import json
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
MODELING_V3_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v3"
PKG_DIR = BASE_DIR / "ml" / "models" / "attribution" / "v1"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase4g"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4g_upstream_artifact_immutability():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    model_hash = calculate_sha256(PKG_DIR / "model.joblib")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    assert model_hash == "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"


def test_phase4g_dataset_v3_structure_and_integrity():
    v3_csv = MODELING_V3_DIR / "feature_dataset_frozen.csv"
    assert v3_csv.exists()

    v3_df = pd.read_csv(v3_csv)
    assert len(v3_df) == 1827
    assert v3_df['date'].min() == "2020-01-01"
    assert v3_df['date'].max() == "2024-12-31"
    assert v3_df['date'].duplicated().sum() == 0

    # Key external features present
    ext_expected = ["rainfall_1d", "rainfall_3d", "pblh_1d", "aod_550_1d", "wind_u_component_1d"]
    for f in ext_expected:
        assert f in v3_df.columns
        assert v3_df[f].isnull().sum() == 0


def test_phase4g_leakage_and_quality_audit_reports():
    qa_report = EXP_DIR / "external_data_quality_report.csv"
    leak_report = EXP_DIR / "leakage_audit.csv"
    src_reg = EXP_DIR / "external_source_registry.csv"

    assert qa_report.exists()
    assert leak_report.exists()
    assert src_reg.exists()

    leak_df = pd.read_csv(leak_report)
    assert (leak_df['detected'] == True).sum() == 0


def test_phase4g_experiment_reports_generated():
    wf_report = EXP_DIR / "walk_forward_results_v3.csv"
    inc_report = EXP_DIR / "incremental_information.csv"
    stat_report = EXP_DIR / "statistical_comparisons.csv"

    assert wf_report.exists()
    assert inc_report.exists()
    assert stat_report.exists()

    inc_df = pd.read_csv(inc_report)
    assert len(inc_df) >= 5
