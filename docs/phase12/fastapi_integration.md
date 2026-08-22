# AtmosIQ Phase 12 — FastAPI Client & Inference Boundary Specification

## 1. Boundary Design

The downstream forecasting microservice is treated as an immutable downstream service. The Spring Boot backend consumes it exclusively via HTTP REST endpoints.

```
Spring Boot (RestClientFastApiInferenceClient)
  --> GET  http://localhost:8000/health
  --> GET  http://localhost:8000/ready
  --> GET  http://localhost:8000/version
  --> POST http://localhost:8000/predict
```

---

## 2. Immutability Guard & Fail-Closed Policy

The `RestClientFastApiInferenceClient` verifies the downstream model release identity:
- **Expected Model ID**: `AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0`
- **Expected SHA-256**: `fdc99f7ca4410f3d577e52718e35c956c97f368cd91a8b7f505ee23824085bac`

If the downstream service reports any other model ID or fails to identify itself, the client immediately throws `ModelIdentityMismatchException` and **FAILS CLOSED**. It will never silently route predictions through an unauthorized or unverified model.

---

## 3. Resilience & Timeout Configuration

- **Connection Timeout**: Externalized via `ATMOSIQ_FASTAPI_CONNECT_TIMEOUT` (Default: $5000\text{ ms}$).
- **Read Timeout**: Externalized via `ATMOSIQ_FASTAPI_READ_TIMEOUT` (Default: $10000\text{ ms}$).
- **No Blind Retries**: Prediction requests are not silently retried to avoid hidden amplification or masking downstream failure states.
