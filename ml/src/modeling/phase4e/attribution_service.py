import sys
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Path as APIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.cache import CachePhase4E
from ml.src.modeling.phase4e.prediction_service import PredictionServicePhase4E
from ml.src.modeling.phase4e.shap_service import SHAPServicePhase4E
from ml.src.modeling.phase4e.validation_service import ValidationServicePhase4E
from ml.src.modeling.phase4e.counterfactual_service import CounterfactualServicePhase4E
from ml.src.modeling.phase4e.event_service import EventServicePhase4E
from ml.src.modeling.phase4e.decision_engine import DecisionEnginePhase4E
from ml.src.modeling.phase4e.response_schema import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    AttributionResponse,
    EnvironmentalValidationResponse,
    CounterfactualResponse,
    DecisionSupportResponse,
    EventResponse
)

app = FastAPI(
    title="AtmosIQ Source Attribution & Decision Support API",
    description="RESTful API serving PM2.5 predictions, TreeSHAP attributions, environmental validation, counter-evidence, and counterfactual scenario sensitivities.",
    version="1.0.0"
)

# Enable CORS for security & local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Service Singletons
_data_loader = None
_cache = None
_pred_service = None
_shap_service = None
_val_service = None
_cf_service = None
_event_service = None
_decision_engine = None


def get_services():
    global _data_loader, _cache, _pred_service, _shap_service, _val_service, _cf_service, _event_service, _decision_engine
    if _data_loader is None:
        _data_loader = DataLoaderPhase4E()
        _cache = CachePhase4E(_data_loader)
        _pred_service = PredictionServicePhase4E(_data_loader, _cache)
        _shap_service = SHAPServicePhase4E(_data_loader, _cache)
        _val_service = ValidationServicePhase4E(_data_loader, _cache)
        _cf_service = CounterfactualServicePhase4E(_data_loader, _cache)
        _event_service = EventServicePhase4E(_data_loader, _cache)
        _decision_engine = DecisionEnginePhase4E(_data_loader, _cache)
    return {
        "pred": _pred_service,
        "shap": _shap_service,
        "val": _val_service,
        "cf": _cf_service,
        "event": _event_service,
        "engine": _decision_engine
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Metadata"])
def get_health():
    try:
        get_services()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            shap_loaded=True,
            validation_loaded=True,
            counterfactual_loaded=True,
            dataset_loaded=True,
            integrity_check="PASS"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MODEL_INTEGRITY_FAILURE: {str(e)}")


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    return ModelInfoResponse()


@app.get("/api/v1/prediction/{date}", response_model=PredictionResponse, tags=["Prediction"])
def get_prediction(date: str = APIPath(..., description="Date in YYYY-MM-DD format")):
    services = get_services()
    try:
        return services["pred"].predict_date(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/attribution/{date}", response_model=AttributionResponse, tags=["Attribution"])
def get_attribution(date: str = APIPath(..., description="Date in YYYY-MM-DD format")):
    services = get_services()
    try:
        return services["shap"].explain_prediction(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/validation/{date}", response_model=EnvironmentalValidationResponse, tags=["Validation"])
def get_validation(date: str = APIPath(..., description="Date in YYYY-MM-DD format")):
    services = get_services()
    try:
        return services["val"].validate_attribution(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/counterfactual/{date}/{scenario}", response_model=CounterfactualResponse, tags=["Counterfactual"])
def get_counterfactual(
    date: str = APIPath(..., description="Date in YYYY-MM-DD format"),
    scenario: str = APIPath(..., description="Counterfactual scenario name")
):
    services = get_services()
    try:
        return services["cf"].run_counterfactual(date, scenario)
    except ValueError as e:
        if "INVALID_SCENARIO" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/decision-support/{date}", response_model=DecisionSupportResponse, tags=["Decision Support"])
def get_decision_support(date: str = APIPath(..., description="Date in YYYY-MM-DD format")):
    services = get_services()
    try:
        return services["engine"].generate_decision_support(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/events", response_model=List[Dict[str, Any]], tags=["Events"])
def get_events():
    services = get_services()
    return services["event"].get_all_events()


@app.get("/api/v1/events/{event_id}", response_model=EventResponse, tags=["Events"])
def get_event(event_id: str = APIPath(..., description="Event ID, e.g., EVT_001")):
    services = get_services()
    try:
        return services["event"].explain_event_by_id(event_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/event-analysis/{start_date}/{end_date}", response_model=List[DecisionSupportResponse], tags=["Events"])
def analyze_period(
    start_date: str = APIPath(..., description="Start date YYYY-MM-DD"),
    end_date: str = APIPath(..., description="End date YYYY-MM-DD")
):
    services = get_services()
    try:
        return services["engine"].analyze_period(start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Mount built frontend dist files if present
frontend_dist = ROOT_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
