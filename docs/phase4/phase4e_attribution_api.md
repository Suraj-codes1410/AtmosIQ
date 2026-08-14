# AtmosIQ — Phase 4E Technical Documentation
## Source Attribution API & Decision Support System Integration

### 1. Overview
Phase 4E transforms the AtmosIQ research pipeline (Phases 3G, 4A, 4B, 4C, 4D) into a clean, programmatically accessible RESTful API and decision-support framework.

### 2. Architecture & Provenance
- **Frozen Model**: `RandomForestRegressor` (450 trees, max depth 9, SHA-256: `55d7f6ab...`)
- **Dataset v2**: 1,827 daily observations (2020-01-01 -> 2024-12-31, SHA-256: `e7645584...`)
- **Feature Vector**: 147 prediction-safe features (zero leakage, lag >= 1d).

### 3. REST API Endpoints
- `GET /api/v1/health`: Health status & artifact integrity check.
- `GET /api/v1/model/info`: Model type, hash, dataset hash, feature registry metadata.
- `GET /api/v1/prediction/{date}`: Observed PM2.5, model prediction, persistence baseline, error.
- `GET /api/v1/attribution/{date}`: Base value, top features, signed/mean-abs group attributions.
- `GET /api/v1/validation/{date}`: Phase 4C independent indicators & explicit counter-evidence conflicts.
- `GET /api/v1/counterfactual/{date}/{scenario}`: Controlled scenario sensitivity prediction & delta.
- `GET /api/v1/decision-support/{date}`: Unified high-level decision report.
- `GET /api/v1/events`: List of 110 extreme pollution episodes.
- `GET /api/v1/events/{event_id}`: Multi-day episode catalog breakdown & counterfactuals.

### 4. Scientific Disclaimer & Non-Causal Safeguards
```text
PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION
```
