# AtmosIQ — Phase 4F Architecture & Technical Audit
## Production Dashboard & Decision Support System Integration Audit

### 1. Executive Context & Scope
Phase 4F establishes a research-grade, user-facing interactive dashboard for AtmosIQ. The dashboard functions as a presentation, exploration, and decision-support interface for the completed Phase 3G production model, Phase 4A reproducibility package, Phase 4B TreeSHAP attributions, Phase 4C environmental validation framework, Phase 4D counterfactual simulation engine, and Phase 4E RESTful API.

### 2. Audit of Existing System Assets

#### A. Locked & Immutable Pipeline Assets
- **Dataset v2**: 1,827 daily observations (`2020-01-01` → `2024-12-31`), 261 raw columns / 147 prediction-safe features.
  - **Immutable SHA-256 Hash**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`
- **Dataset v1**: 731 daily benchmark observations (`2020-01-01` → `2021-12-31`).
  - **Immutable SHA-256 Hash**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Frozen Production Model**: `RandomForestRegressor` (450 trees, max depth 9, 147 features, `min_samples_split=3`, `min_samples_leaf=3`, `max_features=0.5`).
  - **Location**: `ml/models/attribution/v1/model.joblib`
  - **Immutable SHA-256 Hash**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162`

#### B. Phase 4E Backend API Service Audit
FastAPI REST backend (`/api/v1/`) defined in `ml/src/modeling/phase4e/`:
- `GET /api/v1/health`: Checks system health & SHA-256 artifact integrity.
- `GET /api/v1/model/info`: Exposes provenance, feature counts, and hashes.
- `GET /api/v1/prediction/{date}`: Returns observed PM2.5, predicted PM2.5, persistence baseline, and error.
- `GET /api/v1/attribution/{date}`: Returns base value, top features, and signed/mean-abs group SHAP attributions.
- `GET /api/v1/validation/{date}`: Returns independent indicators (fire count, wind speed, PBLH) and counter-evidence conflicts.
- `GET /api/v1/counterfactual/{date}/{scenario}`: Returns model sensitivity $\Delta\hat{y} = f(x_{\text{cf}}) - f(x_{\text{obs}})$ under registered feature interventions.
- `GET /api/v1/decision-support/{date}`: Returns unified high-level decision report.
- `GET /api/v1/events` & `GET /api/v1/events/{event_id}`: Serves 110 cataloged extreme pollution episodes.

### 3. Frontend Architecture Design
- **Technology Stack**: React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Chart.js / Recharts.
- **Location**: `frontend/`
- **Design Philosophy**: Research-grade environmental intelligence platform. Clean scientific visualization, explicit units ($\mu\text{g/m}^3$), transparent uncertainty, explicit non-causal terminology, zero fake real-time data claims.

### 4. Non-Causal Safeguards & Scientific Limitations
$$\text{PREDICTIVE IMPORTANCE} \neq \text{SHAP ATTRIBUTION} \neq \text{COUNTERFACTUAL MODEL RESPONSE} \neq \text{CAUSAL EFFECT} \neq \text{ACTUAL EMISSION CONTRIBUTION}$$
- `pm25_persistence` represents model dependence on prior pollution history, **NOT** an independent physical emission source.
- Counterfactual responses represent model sensitivity under controlled feature interventions, **NOT** physical chemical transport simulations or policy emission-reduction guarantees.
