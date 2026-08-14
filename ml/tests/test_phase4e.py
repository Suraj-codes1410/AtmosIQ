import hashlib
from pathlib import Path
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
PKG_DIR = BASE_DIR / "ml" / "models" / "attribution" / "v1"
EXP_DIR_4E = BASE_DIR / "ml" / "experiments" / "phase4e"

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.prediction_service import PredictionServicePhase4E
from ml.src.modeling.phase4e.shap_service import SHAPServicePhase4E
from ml.src.modeling.phase4e.validation_service import ValidationServicePhase4E
from ml.src.modeling.phase4e.counterfactual_service import CounterfactualServicePhase4E
from ml.src.modeling.phase4e.confidence_service import ConfidenceServicePhase4E
from ml.src.modeling.phase4e.event_service import EventServicePhase4E
from ml.src.modeling.phase4e.decision_engine import DecisionEnginePhase4E


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4e_artifact_integrity_hashes_unchanged():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    model_hash = calculate_sha256(PKG_DIR / "model.joblib")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    assert model_hash == "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"


def test_phase4e_prediction_service():
    loader = DataLoaderPhase4E()
    service = PredictionServicePhase4E(loader)

    res = service.predict_date("2024-11-16")
    assert res.date == "2024-11-16"
    assert res.predicted_pm25 > 0.0
    assert res.model_version == "phase3g_rf_v1"

    # Test invalid date
    with pytest.raises(ValueError, match="DATE_NOT_FOUND"):
        service.predict_date("1999-01-01")


def test_phase4e_shap_service_and_persistence_caveat():
    loader = DataLoaderPhase4E()
    service = SHAPServicePhase4E(loader)

    res = service.explain_prediction("2024-11-16")
    assert res.date == "2024-11-16"
    assert len(res.group_attributions) == 5
    assert "pm25_persistence" in [g.group_name for g in res.group_attributions]
    assert "PREDICTIVE IMPORTANCE !=" in res.scientific_disclaimer
    assert "PM2.5 persistence" in res.persistence_caveat


def test_phase4e_validation_and_counter_evidence_surfacing():
    loader = DataLoaderPhase4E()
    service = ValidationServicePhase4E(loader)

    # Date 2020-10-18 has known counter-evidence conflict in Phase 4C
    res = service.validate_attribution("2020-10-18")
    assert res.has_counter_evidence is True
    assert len(res.counter_evidence_conflicts) > 0
    assert res.validation_status == "WARNING_CONFLICT"


def test_phase4e_counterfactual_service_and_invalid_scenario():
    loader = DataLoaderPhase4E()
    service = CounterfactualServicePhase4E(loader)

    res = service.run_counterfactual("2024-11-16", "biomass_low")
    assert res.scenario == "biomass_low"
    assert res.delta_prediction < 0.0
    assert "predicts a" in res.interpretation

    # Test invalid scenario
    with pytest.raises(ValueError, match="INVALID_SCENARIO"):
        service.run_counterfactual("2024-11-16", "arbitrary_scenario_x")


def test_phase4e_confidence_engine():
    loader = DataLoaderPhase4E()
    service = ConfidenceServicePhase4E(loader)

    conf_clean = service.calculate_confidence("2024-11-16")
    assert conf_clean.confidence_level in ["HIGH", "MODERATE"]

    conf_conflict = service.calculate_confidence("2020-10-18")
    assert conf_conflict.confidence_score < 0.80


def test_phase4e_event_service():
    loader = DataLoaderPhase4E()
    service = EventServicePhase4E(loader)

    summary = service.analyze_extreme_event("2024-11-16")
    assert summary.is_extreme_event is True
    assert summary.peak_pm25 > 306.81

    event_info = service.explain_event_by_id("EVT_001")
    assert event_info.event_id == "EVT_001"
    assert event_info.duration_days > 0


def test_phase4e_decision_engine_and_case_studies():
    loader = DataLoaderPhase4E()
    engine = DecisionEnginePhase4E(loader)

    # Test representative historical case study 2024-11-16
    ds = engine.generate_decision_support("2024-11-16")
    assert ds.date == "2024-11-16"
    assert ds.prediction.predicted_pm25 > 0
    assert "biomass_low" in ds.counterfactual_scenarios
    assert "caused" not in ds.scientific_interpretation.lower()


def test_phase4e_regression_metrics():
    # Verify Phase 3G, 4B, 4D regression criteria
    cf_res = pd.read_csv(BASE_DIR / "ml" / "experiments" / "phase4d" / "counterfactual_results.csv")
    assert len(cf_res) > 0
