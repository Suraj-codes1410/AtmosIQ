import hashlib
from pathlib import Path
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
PKG_DIR = BASE_DIR / "ml" / "models" / "attribution" / "v1"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.decision_engine import DecisionEnginePhase4E


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4f_artifact_integrity_hashes_unchanged():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    model_hash = calculate_sha256(PKG_DIR / "model.joblib")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    assert model_hash == "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"


def test_phase4f_frontend_build_artifacts_exist():
    assert FRONTEND_DIST_DIR.exists(), "Frontend build directory frontend/dist missing!"
    index_file = FRONTEND_DIST_DIR / "index.html"
    assert index_file.exists()
    content = index_file.read_text(encoding="utf-8")
    assert "AtmosIQ" in content


def test_phase4f_non_causal_language_compliance():
    loader = DataLoaderPhase4E()
    engine = DecisionEnginePhase4E(loader)

    report = engine.generate_decision_support("2024-11-16")
    interp = report.scientific_interpretation.lower()

    assert "stubble burning caused" not in interp
    assert "caused today" not in interp
    assert "PREDICTIVE IMPORTANCE !=" in report.scientific_limitations


def test_phase4f_counter_evidence_and_ood_surfacing():
    loader = DataLoaderPhase4E()
    engine = DecisionEnginePhase4E(loader)

    # Date with known conflict 2020-10-18
    report = engine.generate_decision_support("2020-10-18")
    assert report.validation.has_counter_evidence is True
    assert len(report.validation.counter_evidence_conflicts) > 0
    assert report.confidence.confidence_level in ["MODERATE", "LOW"]


def test_phase4f_architecture_audit_document_exists():
    audit_doc = BASE_DIR / "docs" / "phase4" / "phase4f_architecture_audit.md"
    assert audit_doc.exists()
    content = audit_doc.read_text(encoding="utf-8")
    assert "Phase 4F" in content
    assert "55d7f6ab" in content
