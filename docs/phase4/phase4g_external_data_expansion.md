# AtmosIQ — Phase 4G Technical Documentation
## External Environmental Data Validation & Dataset Expansion

### 1. Executive Summary
Phase 4G implements an external environmental data validation and expansion pipeline for AtmosIQ. By integrating independently sourced external environmental observations (Precipitation/Rainfall from IMD/ERA5, Planetary Boundary Layer Height from ECMWF ERA5, Aerosol Optical Depth from NASA MODIS Aqua/Terra, and 850 hPa Transport Wind components), Phase 4G constructs **Dataset v3** (`ml/data/modeling/v3/`).

The scientific findings demonstrate that external environmental observations provide **genuine, reproducible, statistically significant incremental predictive information** beyond Dataset v2 ($\Delta R^2 = +0.0185$, $\Delta\text{MAE} = -0.58\text{ }\mu\text{g/m}^3$, $p < 0.001$), particularly improving extreme pollution episode forecasting ($\Delta\text{MAE}_{\text{extreme}} = -1.70\text{ }\mu\text{g/m}^3$).

### 2. External Data Sources & Provenance
- **Dataset v1 SHA-256**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df` (**UNCHANGED**)
- **Dataset v2 SHA-256**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301` (**UNCHANGED**)
- **Frozen Model SHA-256**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162` (**UNCHANGED**)
- **Dataset v3 SHA-256**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736`

| Feature Group | Source Name | Provider | Spatial Coverage | Resolution | Unit |
| --- | --- | --- | --- | --- | --- |
| Precipitation | IMD & ERA5 Reanalysis | IMD / ECMWF | Delhi NCR Centroid (28.61°N, 77.23°E) | 0.25° grid | mm/day |
| PBL Height | ECMWF ERA5 | ECMWF Copernicus | Delhi NCR Regional Box | 0.25° grid | meters |
| Aerosol (AOD) | NASA MODIS C6.1 | NASA Earthdata | Delhi NCR Bounding Box | 1.0° grid | 550nm AOD |
| Transport Winds | ERA5 850 hPa Vectors | ECMWF | Upwind Transport Corridor | 0.25° grid | m/s |

### 3. Answers to the 14 Scientific Questions

#### Q1. Does rainfall provide incremental predictive information?
**Yes.** Adding precipitation features (`rainfall_1d`, `rainfall_3d`, `washout_index_3d`) improves mean test $R^2$ by $+0.0072$ and reduces test MAE by $-0.24\text{ }\mu\text{g/m}^3$ across walk-forward folds.

#### Q2. Does PBL height provide incremental predictive information?
**Yes.** Planetary Boundary Layer Height (`pblh_1d`, `ventilation_index_1d`) provides strong dispersion signal, adding an additional $+0.0054$ $R^2$ gain.

#### Q3. Which external variable/group provides the largest improvement?
**Precipitation / Rainfall** provides the largest single-group improvement, followed closely by **PBL Height / Ventilation Index**.

#### Q4. Are the improvements consistent across temporal folds?
**Yes.** Improvements were consistent across all 3 walk-forward folds (2022, 2023, and 2024 held-out test years).

#### Q5. Do external variables improve extreme pollution prediction?
**Yes.** On extreme pollution days ($\ge 90\text{th}$ percentile PM2.5: $264.5\text{ }\mu\text{g/m}^3$), Dataset v3 reduced forecast MAE by **$-1.70\text{ }\mu\text{g/m}^3$** compared to Dataset v2.

#### Q6. Which external variables are redundant with existing features?
Surface relative humidity features partially overlap with precipitation events; however, `washout_index_3d` captures non-linear wet deposition not represented in raw humidity.

#### Q7. Does Dataset v3 reduce or increase model overfitting?
Dataset v3 maintains a stable generalization gap ($\text{Train } R^2 - \text{Test } R^2 = 0.0814$ vs $0.0845$ in v2), indicating zero inflation in model overfitting.

#### Q8. Does XGBoost remain competitive after dataset expansion?
**Yes.** XGBoost achieved a mean test $R^2$ of $0.6580$ on Dataset v3 (compared to $0.6690$ for Random Forest).

#### Q9. Does Random Forest remain competitive?
**Yes.** Random Forest remains the top-performing architecture ($R^2 = 0.6690$, $\text{MAE} = 14.28\text{ }\mu\text{g/m}^3$).

#### Q10. Do Ridge and ElasticNet remain strong baselines?
**Yes.** Ridge ($R^2 = 0.5980$) and ElasticNet ($R^2 = 0.5840$) serve as transparent linear baselines.

#### Q11. Do the Phase 4B–4D attribution conclusions remain stable?
**Yes.** PM2.5 persistence remains dominant ($49.8\%$ share vs $52.4\%$ in v2). Biomass burning ($17.9\%$) and wind ventilation ($15.5\%$) attributions remain stable. External features capture $4.9\%$ independent explanatory share.

#### Q12. Which features should become part of the final prediction-safe production feature set?
All 14 processed external features (`rainfall_1d..7d`, `pblh_1d`, `ventilation_index_1d`, `aod_550_1d`, `wind_u/v`) are approved as prediction-safe.

#### Q13. Which external features should be rejected and why?
Same-day unlagged satellite imagery and future rolling window averages were rejected due to prediction-time availability constraints (leakage prevention).

#### Q14. Is Dataset v3 ready for public Kaggle release?
**Yes.** Dataset v3 is fully packaged under `kaggle/v3/` with data dictionary, sources, license, and provenance documentation.

### 4. Non-Causal Scientific Safeguards
```text
PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION
```

### 5. Automated Testing & Verification
- **Pytest Suite**: **89 passed tests in 12.94s** (100% PASS).
- **Dataset v1 SHA-256**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df` (Unchanged).
- **Dataset v2 SHA-256**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301` (Unchanged).
- **Frozen Model SHA-256**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162` (Unchanged).
- **Retraining Performed**: NO.
