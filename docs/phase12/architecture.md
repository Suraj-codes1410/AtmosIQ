# AtmosIQ Phase 12 — Architecture Specification: Spring Boot + Spring AI Orchestration Foundation

## 1. Overview & System Topology

Phase 12 establishes the application orchestration layer for AtmosIQ on top of the certified **AtmosIQ v1.0.0** forecasting system.

The orchestration foundation integrates **Spring Boot 3.3 (Java 21)** and **Spring AI** with strongly typed tool contracts, client boundaries, provenance propagation, and strict tool allowlisting.

```mermaid
graph TD
    subgraph ClientLayer ["Client Application Layer"]
        User["User / API Client"]
    end

    subgraph SpringBootOrchestration ["Spring Boot 3.3 Orchestration Engine"]
        Filter["CorrelationIdFilter (X-Correlation-ID, X-Request-ID)"]
        Controller["ForecastController / MonitoringController"]
        OrchestrationSvc["OrchestrationService"]
        Registry["ToolRegistry (Allowlist Enforcer)"]
        Metrics["OrchestrationMetrics & Telemetry"]
        
        subgraph SpringAILayer ["Spring AI Tool Integration"]
            SpringAiConf["SpringAiConfig (@Bean Tool Definitions)"]
            ForecastTool["forecast_pm25 (ForecastTool)"]
            HealthTool["check_service_health (HealthTool)"]
            MetadataTool["get_model_metadata (ModelMetadataTool)"]
        end
    end

    subgraph ClientBoundary ["Client Boundary"]
        FastApiClient["FastApiInferenceClient (RestClient)"]
        ImmutabilityGuard["Immutability Guard (Model ID Verification)"]
    end

    subgraph MLServiceBoundary ["Certified Downstream Inference Service (Immutable)"]
        FastApiApp["FastAPI REST Service (/health, /ready, /version, /predict)"]
        TCNModel["Certified TCN v1.0.0 Model (849 Params, 25% CAL-07)"]
        UncertaintyEngine["Conformal Prediction Bounds (90%: ±95.66 µg/m³)"]
    end

    User -->|HTTP POST /api/v1/forecast| Filter
    Filter --> Controller
    Controller --> OrchestrationSvc
    OrchestrationSvc --> Registry
    Registry --> ForecastTool
    ForecastTool --> FastApiClient
    FastApiClient --> ImmutabilityGuard
    ImmutabilityGuard --> FastApiApp
    FastApiApp --> TCNModel
    TCNModel --> UncertaintyEngine

    UncertaintyEngine -.->|Calibrated Predictions & Conformal Bounds| FastApiApp
    FastApiApp -.->|Raw JSON Forecasts| FastApiClient
    FastApiClient -.->|Typed DTOs| ForecastTool
    ForecastTool -.->|Provenance & Uncertainty DTOs| OrchestrationSvc
    OrchestrationSvc -.->|Validated Application Response| Controller
    Controller -.->|HTTP 200 OK Response| User
```

---

## 2. Core Architectural Roles

- **Spring Boot**: Hosts application REST controllers, global exception handling, correlation ID tracking, metrics, and application DTOs.
- **Spring AI**: Exposes structured function-calling tool contracts (`forecast_pm25`, `check_service_health`, `get_model_metadata`) with documented JSON schemas.
- **FastAPI Client**: Handles HTTP communication to the certified inference microservice with timeouts and error mapping.
- **TCN v1.0.0**: The certified downstream deep learning model (849 parameters, $W=14, D=35$). The ML model and weights remain 100% frozen and immutable.
