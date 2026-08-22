# AtmosIQ Phase 12 — Controlled Tool Contracts Specification

## 1. Registered Tool Inventory

| Tool Name | Class | Contract Description | Parameters |
| :--- | :--- | :--- | :--- |
| `forecast_pm25` | `ForecastTool` | Executes calibrated PM2.5 forecast with 90% conformal intervals using certified TCN model | `records`: List of $\ge 14$ rows of 35 features, `correlationId` |
| `check_service_health` | `HealthTool` | Checks operational health and model readiness of downstream inference microservice | None |
| `get_model_metadata` | `ModelMetadataTool` | Retrieves architectural specifications, parameter counts (849), and SHA-256 identity | None |

---

## 2. Tool Contract Rules & Invariants

1. **Non-Modification Rule**: Tools must NEVER alter model weights, calibration offsets ($-5.06\text{ }\mu\text{g/m}^3$), or conformal prediction interval half-widths ($\pm 95.66\text{ }\mu\text{g/m}^3$).
2. **Schema Invariance**: The input sequence MUST contain all 35 certified prediction-safe features for at least 14 sequential daily records ($W = 14$).
3. **Immutability Verification**: The forecast tool validates that the downstream service reports `AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0` before propagating results.
4. **Physical Disclaimer**: Every forecast item returned by `forecast_pm25` explicitly attaches:
   `"physicalDisclaimer": "PREDICTION INTERVAL != GUARANTEED PHYSICAL UNCERTAINTY"`
