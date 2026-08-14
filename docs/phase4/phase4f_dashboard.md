# AtmosIQ — Phase 4F Technical Documentation
## Production Dashboard & Decision Support Interface

### 1. Implementation Summary
Phase 4F implements a research-grade, user-facing interactive dashboard for AtmosIQ using React 18, TypeScript, Vite, and Tailwind CSS, integrated with the Phase 4E FastAPI backend.

The dashboard allows atmospheric scientists, environmental analysts, and policy researchers to:
1. Understand PM2.5 forecasts for any date in Dataset v2 (2020-01-01 to 2024-12-31).
2. Inspect TreeSHAP source-group and feature-level attributions.
3. Validate attributions against independent observational indicators (satellite fire counts, wind speed, PBLH).
4. Identify counter-evidence conflicts surfaced explicitly without suppression.
5. Simulate controlled counterfactual feature interventions ($\Delta\hat{y} = f(x_{\text{cf}}) - f(x_{\text{obs}})$).
6. Explore 110 extreme pollution episodes.
7. Analyze 5-year historical trends and seasonal regime shifts.

### 2. Architecture & Data Flow
```text
  ┌─────────────────────────────────────────────────────────────┐
  │                 Phase 4F React + TS Dashboard               │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ HTTP / REST
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Phase 4E FastAPI Router (/api/v1/)          │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Phase 4E Service & Cache Layer              │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Locked Upstream Assets: Dataset v2, Frozen RF, SHAP, Val   │
  └─────────────────────────────────────────────────────────────┘
```

### 3. Key Implemented UI Views & Components
- **`Header.tsx`**: Title, navigation tabs, date picker selector (`2020-01-01` → `2024-12-31`), and live system health badge.
- **`PredictionCard.tsx`**: Observed surface PM2.5, AtmosIQ model prediction, persistence baseline ($t-1\text{d}$), prediction error, and AQI pollution category.
- **`AttributionPanel.tsx`**: Group attributions for `pm25_persistence`, `biomass_burning`, `wind_ventilation`, `meteorology`, `calendar_seasonal` with toggle between Signed SHAP ($\mu\text{g/m}^3$) and Mean Absolute SHAP Share (%).
- **`FeatureImportanceChart.tsx`**: Top positive and negative feature attributions with expandable technical additivity proof ($f(x) = \text{base\_value} + \sum \text{SHAP}_j$).
- **`EnvironmentalEvidence.tsx`**: Independent indicators (satellite MODIS/VIIRS fire hotspots, surface wind speed, boundary layer height).
- **`CounterEvidenceAlert.tsx`**: Prominent warning banner displayed ONLY when counter-evidence conflict exists in `attribution_conflicts.csv`.
- **`CounterfactualSimulator.tsx`**: Interactive scenario selector (`biomass_low`, `wind_dispersion`, `combined_all_favorable`, etc.) with OOD warnings and non-causal wording.
- **`ConfidenceBadge.tsx`**: Evidence-based `HIGH`, `MODERATE`, `LOW`, `INVALID` confidence badge.
- **`EventExplorer.tsx` & `EventDetailModal.tsx`**: Catalog of 110 extreme pollution episodes filterable by year, season, peak PM2.5, and dominant group.
- **`PollutionTimeline.tsx`**: 5-year historical timeline (2020–2024) highlighting extreme pollution episodes ($\ge 306.81\text{ }\mu\text{g/m}^3$).
- **`SeasonalAnalysis.tsx`**: Regime analysis for Post-Monsoon, Winter, Summer, and Monsoon seasons.
- **`MethodologyPage.tsx`**: Interactive sitemap and pipeline specifications.
- **`ScientificDisclaimer.tsx`**: Mandatory non-causal language banner.

### 4. Non-Causal Language Safeguards
$$\text{PREDICTIVE IMPORTANCE} \neq \text{SHAP ATTRIBUTION} \neq \text{COUNTERFACTUAL MODEL RESPONSE} \neq \text{CAUSAL EFFECT} \neq \text{ACTUAL EMISSION CONTRIBUTION}$$
- **PM2.5 Persistence**: Clarified as prior atmospheric state history, not an emission source.
- **Counterfactual Response**: Framed as frozen model sensitivity under simulated feature inputs.

### 5. Verified Immutable Hashes
- **Dataset v2 SHA-256**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301` (**UNCHANGED**)
- **Dataset v1 SHA-256**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df` (**UNCHANGED**)
- **Frozen Model SHA-256**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162` (**UNCHANGED**)

### 6. Automated Testing & Verification
- **Pytest Suite**: 85 passed tests in 12.67s.
- **Frontend Build**: React 18 production bundle compiled cleanly in 2.25s under `frontend/dist`.
- **Retraining Performed**: NO.
- **Frozen Artifacts Modified**: NO.

### 7. Phase 4G Readiness
The architecture is 100% ready for Phase 4G expansion (additional geographic regions, live operational data ingestion, and multi-city forecasting).
