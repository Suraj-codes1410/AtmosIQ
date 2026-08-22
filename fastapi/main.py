"""
AtmosIQ Certified Production FastAPI Inference Microservice.

Serves certified AtmosIQ v1.0.0 TCN forecasting model via HTTP REST endpoints.
Uses Phase10DDeploymentService directly on top of frozen release bundle.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.src.modeling.phase10d.deployment import (
    Phase10DDeploymentService,
    ServiceContractException,
)

logger = logging.getLogger("atmosiq_fastapi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "ml/experiments/phase10d_release/release_bundle"

app = FastAPI(
    title="AtmosIQ Certified Production Inference Microservice",
    version="1.0.0",
    description="Certified REST Inference API serving AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0",
)

# Initialize certified deployment service
deployment_service = Phase10DDeploymentService(BUNDLE_DIR)


class PredictPayload(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="Sequence of at least 14 daily rows containing all 35 prediction-safe features",
    )


@app.get("/health", tags=["Monitoring"])
def get_health() -> Dict[str, Any]:
    """Health liveness probe endpoint."""
    return deployment_service.health_endpoint()


@app.get("/ready", tags=["Monitoring"])
def get_ready() -> Dict[str, Any]:
    """Readiness probe endpoint verifying model, scaler, calibration, and conformal bounds."""
    return deployment_service.readiness_endpoint()


@app.get("/version", tags=["Governance"])
def get_version() -> Dict[str, Any]:
    """Model version and cryptographic identity endpoint."""
    return deployment_service.version_endpoint()


@app.post("/predict", tags=["Inference"])
def post_predict(payload: PredictPayload) -> Dict[str, Any]:
    """
    Executes PM2.5 forecasting over input sequence.
    Applies StandardScaler transform, TCN inference, runtime calibration (-5.06 µg/m³),
    and 90% conformal uncertainty interval computation (±95.66 µg/m³).
    """
    try:
        raw_dict = payload.model_dump()
        result = deployment_service.predict_endpoint(raw_dict)
        return result
    except ServiceContractException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference Contract Violation: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference service error",
        )
