import json
import joblib
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
PKG_DIR = BASE_DIR / "ml" / "models" / "attribution" / "v1"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase4a"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4a_model_file_exists_and_can_be_loaded():
    model_joblib = PKG_DIR / "model.joblib"
    assert model_joblib.exists(), f"Model file missing: {model_joblib}"

    model = joblib.load(model_joblib)
    assert model is not None, "Failed to load model.joblib"


def test_phase4a_model_and_dataset_hashes():
    model_joblib = PKG_DIR / "model.joblib"
    manifest_json = PKG_DIR / "model_manifest.json"

    assert manifest_json.exists()
    with open(manifest_json, "r") as f:
        manifest = json.load(f)

    model_hash = calculate_sha256(model_joblib)
    assert model_hash == manifest["model_sha256"], "Model SHA-256 discrepancy!"

    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    assert v2_hash == manifest["dataset_sha256"]


def test_phase4a_feature_ordering_and_availability():
    feat_reg_file = PKG_DIR / "feature_registry.csv"
    assert feat_reg_file.exists()

    feat_reg = pd.read_csv(feat_reg_file).sort_values("model_feature_order")
    features = feat_reg["feature_name"].tolist()

    assert len(features) == 147
    assert "pm25" not in features
    assert "date" not in features

    avail_df = pd.read_csv(MODELING_V2_DIR / "feature_availability.csv")
    same_day_cols = set(avail_df[avail_df["availability_class"] == "SAME_DAY_FEATURE"]["feature_name"])

    for feat in features:
        assert feat not in same_day_cols, f"Same-day feature '{feat}' leaked into attribution package!"


def test_phase4a_attribution_group_mappings():
    attr_groups_file = PKG_DIR / "attribution_groups.csv"
    assert attr_groups_file.exists()

    attr_df = pd.read_csv(attr_groups_file)
    assert len(attr_df) == 147
    assert (attr_df["attribution_group"] != "unmapped").all(), "Unmapped features exist in attribution groups!"


def test_phase4a_model_prediction_and_reproducibility():
    model = joblib.load(PKG_DIR / "model.joblib")
    feat_reg = pd.read_csv(PKG_DIR / "feature_registry.csv").sort_values("model_feature_order")
    features = feat_reg["feature_name"].tolist()

    df = pd.read_csv(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    X = df[features]

    assert X.isnull().sum().sum() == 0
    assert np.isinf(X.values).sum() == 0

    preds = model.predict(X)
    assert len(preds) == 1827
    assert np.isnan(preds).sum() == 0
    assert np.isinf(preds).sum() == 0


def test_phase4a_manifests_and_checksums():
    checksums_file = PKG_DIR / "checksums.txt"
    manifest_file = PKG_DIR / "model_manifest.json"

    assert checksums_file.exists()
    assert manifest_file.exists()

    with open(manifest_file, "r") as f:
        m = json.load(f)
    assert m["attribution_ready"] is True
