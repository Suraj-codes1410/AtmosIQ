# AtmosIQ — Phase 7A Technical Specification
# Physics-Informed Synthetic Data Design, Repository Audit & Implementation Blueprint

---

## 1. Executive Summary

**Phase 7A** establishes the scientific, statistical, physical, and architectural specification for **Physics-Informed Synthetic Data Generation** within the AtmosIQ research platform.

AtmosIQ has achieved a frozen production point-forecasting baseline (**`MODEL_V3_PRODUCTION`**, $R^2 = 0.9497$, $\text{MAE} = 17.02\,\mu\text{g/m}^3$) and a calibrated prediction-interval and decision-support layer (**`ATMOSIQ_DECISION_SUPPORT v1.0.0`**, empirical coverage = $89.78\%$ at $90\%$ nominal level, $89.01\%$ on severe episodes $\ge 250\,\mu\text{g/m}^3$).

While the current 5-year observational dataset ($N = 1,827$ daily records, 2020–2024) is sufficient for historical validation, training future deep sequence models (Phase 9) and fine-grained policy intelligence agents (Phase 10) requires a substantially expanded training corpus (**Phase 8**). However, standard naive data augmentation (such as random Gaussian noise, unconstrained SMOTE, or independent marginal shuffling) destroys the physical, temporal, and meteorological co-dependencies inherent in atmospheric boundary-layer physics.

**The objective of Phase 7 is to design a Physics-Informed Synthetic Data Engine that generates realistic, physically bounded, temporally coherent, and regime-stratified daily atmospheric observation sequences without violating physical laws, causing distribution collapse, or contaminating locked evaluation data.**

Phase 7A delivers the formal technical blueprint, repository audit, feature inventory, physical relationship matrix, generation architecture selection, validation framework, and Go/No-Go criteria required before implementing synthetic generation in Phase 7B.

---

## 2. Current AtmosIQ Baseline

The validated, immutable production baseline established across Phases 1–6F is defined as follows:

```
                                  ATMOSIQ PRODUCTION STACK (v1.0.0)
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Point Forecasting Engine: MODEL_V3_PRODUCTION (RandomForest, 400 trees, 35 prediction-safe)  │
│    • Walk-Forward Evaluation (2022–2024, N=1,096 held-out days):                                 │
│      - Out-of-sample R²: 0.9497                                                                  │
│      - MAE: 17.02 µg/m³ | RMSE: 26.61 µg/m³                                                      │
│      - Pearson r: 0.9754 | Spearman ρ: 0.9632                                                    │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Production Uncertainty Layer: normalized_conformal v1.0.0                                     │
│    • 80% Nominal Target: 80.66% Empirical Coverage (MPIW: 50.85 µg/m³)                          │
│    • 90% Nominal Target: 89.78% Empirical Coverage (MPIW: 68.77 µg/m³, Winkler: 88.22)          │
│    • 95% Nominal Target: 95.71% Empirical Coverage (MPIW: 87.80 µg/m³)                          │
│    • Severe Pollution (≥250 µg/m³) 90% Coverage: 89.01%                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Interpretability & Counterfactual Layer (Phase 6E):                                           │
│    • TreeSHAP Additivity: 100.0% Pass Rate                                                       │
│    • 6 Environmental Process Groups (Persistence, Biomass, Wind, Meteorology, Ext., Calendar)   │
│    • 8 Predefined Counterfactual Intervention Scenarios                                          │
│    • OOD Distance Scaling: Spearman ρ(OOD, σ_Δ) = +0.7637 (p < 10⁻¹⁵)                            │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Unified Decision Support Service: ATMOSIQ_DECISION_SUPPORT v1.0.0                             │
│    • 3-Tier Deterministic Reliability: HIGH_RELIABILITY, MODERATE_RELIABILITY, HIGH_UNCERTAINTY  │
│    • Traceable Atmospheric Evidence & Counter-Evidence Synthesis                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Audit

A comprehensive audit of the AtmosIQ codebase was conducted to establish exact artifact locations, schemas, and verified cryptographic hashes:

```
AtmosIQ Repository Root
├── ml/
│   ├── data/
│   │   ├── raw/ (openaq_delhi_raw.csv, nasa_firms_raw.csv, open_meteo_raw.csv, calendar_raw.csv)
│   │   └── modeling/
│   │       ├── v1/feature_dataset_frozen.csv (SHA-256: c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df)
│   │       ├── v2/feature_dataset_frozen.csv (SHA-256: e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301)
│   │       └── v3/feature_dataset_frozen.csv (SHA-256: 78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736)
│   ├── models/
│   │   ├── attribution/v1/model.joblib (SHA-256: 55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162)
│   │   └── production/v3/
│   │       ├── model.joblib (SHA-256: 9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210)
│   │       └── feature_registry.csv (35 approved prediction-safe features)
│   ├── uncertainty/production/v1/ (uncertainty_method.json, calibration_artifacts.json, README.md)
│   ├── decision_support/production/v1/ (decision_support_schema.json, decision_rules.json, README.md)
│   ├── src/
│   │   ├── features/ (calendar, fire, interaction, pollution, time, weather)
│   │   └── modeling/ (phase1 through phase6f runners and engines)
│   ├── experiments/ (phase4a–phase4j, phase6a–phase6f)
│   └── tests/ (163 passing unit tests across all phases)
└── docs/ (phase1, phase2, phase3, phase4, phase6 documentation)
```

### Audit Findings & Immutability Verification:
1. **Upstream Hashes**: 100% verified. All frozen datasets and production models match their exact recorded hashes.
2. **Feature Registry**: Exactly 35 approved prediction-safe features.
3. **Target Variable**: `pm25` (ambient daily average concentration in $\mu\text{g/m}^3$).
4. **Temporal Extent**: 1,827 consecutive calendar days from `2020-01-01` to `2024-12-31` (731 training days: 2020–2021; 1,096 held-out evaluation days: 2022–2024).

---

## 4. Dataset Schema Analysis

Dataset v3 (`ml/data/modeling/v3/feature_dataset_frozen.csv`) contains 1,827 rows and 275 columns, structured as:

1. **Primary Temporal Key**: `date` (ISO-8601 string `YYYY-MM-DD`, monotonic daily increment, 0 duplicates, 0 missing).
2. **Target Variable**: `pm25` (Float64, range: $[21.99, 394.76]\,\mu\text{g/m}^3$, mean: $142.86\,\mu\text{g/m}^3$).
3. **Core Meteorological Variables**: Surface temperature (`temperature_c`), relative humidity (`humidity_pct`), 10m wind speed (`wind_speed_kmh`), wind direction (`wind_direction_deg`), surface pressure (`pressure_hpa`), daily precipitation (`precipitation_mm`).
4. **Satellite & Atmospheric Physics Inputs**:
   - NASA FIRMS MODIS/VIIRS: `fire_hotspot_count`, `high_confidence_fire_count`, `mean_fire_brightness`, state-level counts (Punjab, Haryana, Rajasthan, Delhi NCR).
   - ERA5 Reanalysis: Planetary Boundary Layer Height (`pblh_1d`, `pblh_min_1d`, `pblh_roll_mean_3d`), Zonal/Meridional winds (`wind_u_component_1d`, `wind_v_component_1d`), Ventilation Index (`ventilation_index_1d = wind_speed * pblh`).
   - Satellite MODIS AOD: Aerosol Optical Depth at 550nm (`aod_550_1d`).
5. **Engineered Lags & Rolling Aggregations**: 1d, 2d, 3d, 7d, 14d, 30d lags, means, medians, minima, maxima, standard deviations, and variances.
6. **Calendar & Domain Encodings**: `is_stubble_season`, `is_weekend`, `is_holiday`, `festival_window`, `is_winter`, `is_summer`, `is_monsoon`, `is_post_monsoon`.

---

## 5. 35-Feature Production Inventory

Below is the complete inventory of all 35 features approved in `ml/models/production/v3/feature_registry.csv`:

| # | Feature Name | Dtype | Unit | Type | Expected Range | Physical Meaning & Semantics | Synthetic Recommendation |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | `pm25_lag_1d` | Float | $\mu\text{g/m}^3$ | Lag (1d) | $[20.0, 450.0]$ | 1-day prior ambient PM2.5 baseline | Primary state variable |
| **2** | `pm25_lag_2d` | Float | $\mu\text{g/m}^3$ | Lag (2d) | $[20.0, 450.0]$ | 2-day prior ambient PM2.5 baseline | Derived from trajectory history |
| **3** | `pm25_lag_3d` | Float | $\mu\text{g/m}^3$ | Lag (3d) | $[20.0, 450.0]$ | 3-day prior ambient PM2.5 baseline | Derived from trajectory history |
| **4** | `pm25_lag_7d` | Float | $\mu\text{g/m}^3$ | Lag (7d) | $[20.0, 450.0]$ | 7-day prior ambient PM2.5 baseline | Derived from trajectory history |
| **5** | `pm25_roll_mean_3d` | Float | $\mu\text{g/m}^3$ | Rolling | $[20.0, 400.0]$ | 3-day moving average accumulation | Deterministic rolling function |
| **6** | `pm25_roll_mean_7d` | Float | $\mu\text{g/m}^3$ | Rolling | $[25.0, 380.0]$ | 7-day moving average accumulation | Deterministic rolling function |
| **7** | `pm25_roll_mean_14d` | Float | $\mu\text{g/m}^3$ | Rolling | $[30.0, 350.0]$ | 14-day moving average baseline | Deterministic rolling function |
| **8** | `pm25_roll_std_7d` | Float | $\mu\text{g/m}^3$ | Rolling | $[0.0, 150.0]$ | 7-day exposure volatility/variance | Deterministic rolling function |
| **9** | `pm25_roll_max_7d` | Float | $\mu\text{g/m}^3$ | Rolling | $[40.0, 450.0]$ | 7-day peak exposure level | Deterministic rolling function |
| **10** | `pm25_roll_min_7d` | Float | $\mu\text{g/m}^3$ | Rolling | $[20.0, 350.0]$ | 7-day minimum background level | Deterministic rolling function |
| **11** | `temperature_c_lag_1d` | Float | °C | Lag (1d) | $[5.0, 48.0]$ | Surface ambient temperature lag | Primary meteorological driver |
| **12** | `temperature_c_roll_mean_3d` | Float | °C | Rolling | $[6.0, 45.0]$ | 3-day mean thermal trend | Deterministic rolling function |
| **13** | `temperature_c_roll_min_3d` | Float | °C | Rolling | $[4.0, 42.0]$ | 3-day minimum thermal inversion proxy | Deterministic rolling function |
| **14** | `humidity_pct_lag_1d` | Float | % | Lag (1d) | $[10.0, 100.0]$ | Surface relative humidity lag | Hygroscopic growth driver |
| **15** | `humidity_pct_roll_mean_3d` | Float | % | Rolling | $[12.0, 98.0]$ | 3-day mean relative humidity | Deterministic rolling function |
| **16** | `humidity_pct_roll_max_7d` | Float | % | Rolling | $[15.0, 100.0]$ | 7-day maximum humidity indicator | Deterministic rolling function |
| **17** | `wind_speed_kmh_lag_1d` | Float | km/h | Lag (1d) | $[2.0, 45.0]$ | 10m surface wind speed lag | Mechanical dispersion driver |
| **18** | `wind_speed_kmh_roll_mean_3d` | Float | km/h | Rolling | $[3.0, 38.0]$ | 3-day mean surface wind speed | Deterministic rolling function |
| **19** | `wind_u_component_1d` | Float | m/s | Physical | $[-15.0, +15.0]$ | Zonal wind component (East-West) | Coupled vector generation |
| **20** | `wind_v_component_1d` | Float | m/s | Physical | $[-15.0, +15.0]$ | Meridional wind component (North-South) | Coupled vector generation |
| **21** | `is_stubble_season` | Int | Binary | Flag | $\{0, 1\}$ | Oct 15 – Nov 30 burning window | Deterministic calendar flag |
| **22** | `fire_hotspot_count_lag_1d` | Float | Count | Lag (1d) | $[0.0, 2500.0]$ | Daily satellite active fire detections | Conditional Poisson/NegBinom |
| **23** | `fire_hotspot_count_roll_mean_3d` | Float | Count | Rolling | $[0.0, 1800.0]$ | 3-day mean upstream fire activity | Deterministic rolling function |
| **24** | `fire_hotspot_count_roll_mean_7d` | Float | Count | Rolling | $[0.0, 1500.0]$ | 7-day cumulative fire activity | Deterministic rolling function |
| **25** | `upwind_stubble_quadrant_1d` | Float | Score | Physical | $[0.0, 50.0]$ | NW corridor stubble transport score | Coupled fire $\times$ wind vector |
| **26** | `rainfall_1d` | Float | mm | Lag (1d) | $[0.0, 250.0]$ | Daily cumulative rainfall | Zero-inflated continuous |
| **27** | `rainfall_3d` | Float | mm | Rolling | $[0.0, 400.0]$ | 3-day cumulative rainfall | Deterministic rolling sum |
| **28** | `rain_event_1d` | Int | Binary | Physical | $\{0, 1\}$ | Binary rain indicator ($\text{Rain} \ge 1\,\text{mm}$) | Derived: $\mathbb{I}(\text{Rain} \ge 1.0)$ |
| **29** | `washout_index_3d` | Float | Index | Physical | $[0.0, 10.0]$ | Aerosol wet deposition washout | Deterministic function |
| **30** | `pblh_1d` | Float | m | Physical | $[300.0, 2500.0]$ | Planetary Boundary Layer Height | Thermal/diurnal physical state |
| **31** | `pblh_min_1d` | Float | m | Physical | $[150.0, 1800.0]$ | Nocturnal inversion layer depth | Coupled with temperature & season |
| **32** | `pblh_roll_mean_3d` | Float | m | Rolling | $[350.0, 2400.0]$ | 3-day mean boundary layer height | Deterministic rolling function |
| **33** | `ventilation_index_1d` | Float | $\text{m}^2/\text{s}$ | Physical | $[500.0, 25000.0]$ | Ventilation capacity: $\text{Wind} \times \text{PBLH}$ | Derived: $\text{ws} \times \text{pblh}$ |
| **34** | `aod_550_1d` | Float | Unitless | Physical | $[0.05, 2.50]$ | MODIS Aerosol Optical Depth (550nm) | Coupled with column PM2.5 |
| **35** | `festival_window` | Int | Binary | Flag | $\{0, 1\}$ | Diwali / Festival anthropogenic window | Deterministic calendar flag |

---

## 6. Environmental Group Mapping

The 35 features are deterministically partitioned into 6 established process groups:

```mermaid
graph TD
    subgraph Groups [6 Environmental Process Groups]
        G1[1. pm25_persistence: 10 features]
        G2[2. wind_ventilation: 8 features]
        G3[3. meteorology: 6 features]
        G4[4. biomass_burning: 5 features]
        G5[5. external_environmental: 5 features]
        G6[6. calendar_seasonal: 1 feature]
    end
```

1. **`pm25_persistence` (10 features)**: `pm25_lag_1d`, `pm25_lag_2d`, `pm25_lag_3d`, `pm25_lag_7d`, `pm25_roll_mean_3d`, `pm25_roll_mean_7d`, `pm25_roll_mean_14d`, `pm25_roll_std_7d`, `pm25_roll_max_7d`, `pm25_roll_min_7d`.
2. **`wind_ventilation` (8 features)**: `wind_speed_kmh_lag_1d`, `wind_speed_kmh_roll_mean_3d`, `wind_u_component_1d`, `wind_v_component_1d`, `pblh_1d`, `pblh_min_1d`, `pblh_roll_mean_3d`, `ventilation_index_1d`.
3. **`meteorology` (6 features)**: `temperature_c_lag_1d`, `temperature_c_roll_mean_3d`, `temperature_c_roll_min_3d`, `humidity_pct_lag_1d`, `humidity_pct_roll_mean_3d`, `humidity_pct_roll_max_7d`.
4. **`biomass_burning` (5 features)**: `is_stubble_season`, `fire_hotspot_count_lag_1d`, `fire_hotspot_count_roll_mean_3d`, `fire_hotspot_count_roll_mean_7d`, `upwind_stubble_quadrant_1d`.
5. **`external_environmental` (5 features)**: `rainfall_1d`, `rainfall_3d`, `rain_event_1d`, `washout_index_3d`, `aod_550_1d`.
6. **`calendar_seasonal` (1 feature)**: `festival_window`.

---

## 7. Physics Relationship Matrix

Atmospheric co-dependencies must be classified by constraint type so synthetic generators do not generate physically impossible states:

| Relationship Pair | Relationship Nature | Classification | Formal Constraint / Physical Rule |
| :--- | :--- | :---: | :--- |
| **$\text{PM}_{2.5} \ge 0$** | Mass non-negativity | `HARD_PHYSICAL_CONSTRAINT` | Lower bound $y \ge 0.0\,\mu\text{g/m}^3$ everywhere |
| **$\text{Ventilation Index}$** | Hydrodynamic definition | `HARD_PHYSICAL_CONSTRAINT` | $\text{VI} \equiv \text{Wind Speed} \times \text{PBLH} \ge 0$ |
| **$\text{Rain Event Indicator}$** | Logical thresholding | `HARD_PHYSICAL_CONSTRAINT` | $\text{rain\_event\_1d} \equiv \mathbb{I}(\text{rainfall\_1d} \ge 1.0\,\text{mm})$ |
| **$\text{Rolling Bounds}$** | Mathematical aggregation | `HARD_PHYSICAL_CONSTRAINT` | $\min_{k}(x) \le \text{mean}(x) \le \max_{k}(x)$ |
| **$\text{Wind Vectors}$** | Trigonometric consistency | `HARD_PHYSICAL_CONSTRAINT` | $\sqrt{u^2 + v^2} \approx \text{wind\_speed}_{\text{m/s}}$ |
| **$\text{Rainfall} \to \text{PM}_{2.5}$** | Wet scavenging / washout | `SOFT_PHYSICAL_PRIOR` | $\text{Rain} > 25\,\text{mm} \implies \Delta \text{PM}_{2.5} < 0$ (95% empirical prob) |
| **$\text{PBLH} \to \text{PM}_{2.5}$** | Mixing volume entrapment | `SOFT_PHYSICAL_PRIOR` | Low PBLH ($\le 400\,\text{m}$) traps surface emissions; expands variance |
| **$\text{Stubble Fires} \to \text{PM}_{2.5}$** | Upwind aerosol transport | `REGIME_CONDITIONAL_RELATIONSHIP` | High fire impact active only when `is_stubble_season == 1` & wind from NW |
| **$\text{Temperature} \to \text{PBLH}$** | Thermal convective mixing | `STATISTICAL_RELATIONSHIP` | Summer high temp ($>35^\circ\text{C}$) correlates with deep PBLH ($>1500\,\text{m}$) |
| **$\text{PM}_{2.5}(t) \to \text{PM}_{2.5}(t-1)$** | Atmospheric inertia | `TEMPORAL_RELATIONSHIP` | Lag-1 autocorrelation $\rho_1 \approx 0.88$; episode memory decay |

---

## 8. Temporal Dependency Analysis

Atmospheric pollutants do not behave as independent identically distributed (i.i.d.) draws. The observational record exhibits strong temporal autocorrelation:

1. **Autoregressive Decay**:
   - $\rho(\text{lag-1d}) = +0.884$
   - $\rho(\text{lag-2d}) = +0.762$
   - $\rho(\text{lag-3d}) = +0.658$
   - $\rho(\text{lag-7d}) = +0.485$
2. **Multi-Day Episode Stagnation**:
   - Severe pollution episodes ($\text{PM}_{2.5} \ge 250\,\mu\text{g/m}^3$) persist for an average of **$4.2 \pm 2.1$ consecutive days**.
   - Clean monsoon windows ($\text{PM}_{2.5} < 60\,\mu\text{g/m}^3$) persist for **$12.5 \pm 6.8$ consecutive days**.
3. **Implication for Synthesis**: Generating isolated single rows independently creates discontinuous jumps that destroy rolling statistics. **Synthetic data MUST be generated as sequential trajectories (blocks of 14–30 contiguous days)** where lag and rolling features are deterministically computed along the trajectory.

---

## 9. Pollution-Regime Analysis

AtmosIQ categorizes ambient air quality into 4 distinct operational regimes:

| Regime Name | $\text{PM}_{2.5}$ Range | Observational Frequency ($N=1,827$) | Regime Characteristics & Dynamics |
| :--- | :---: | :---: | :--- |
| **Low** | $< 60\,\mu\text{g/m}^3$ | 572 days (31.3%) | Clean air, high PBLH, wet monsoon washout, low volatility ($\sigma \approx 9.42\,\mu\text{g/m}^3$) |
| **Moderate** | $60–120\,\mu\text{g/m}^3$ | 501 days (27.4%) | Typical summer/transitional baseline, moderate ventilation ($\sigma \approx 14.85\,\mu\text{g/m}^3$) |
| **High** | $120–250\,\mu\text{g/m}^3$ | 448 days (24.5%) | Poor air quality, early winter inversions, moderate fire activity ($\sigma \approx 28.12\,\mu\text{g/m}^3$) |
| **Extreme** | $\ge 250\,\mu\text{g/m}^3$ | 306 days (16.8%) | Severe smog episodes, shallow nocturnal inversions, peak stubble fires ($\sigma \approx 44.81\,\mu\text{g/m}^3$) |

---

## 10. Seasonal Structure

The local meteorology of the Indo-Gangetic Plain follows 4 pronounced seasons:

1. **Winter (Dec 1 – Feb 28)**: Shallow boundary layers ($300–600\,\text{m}$), nocturnal radiation inversions, calm surface winds ($<10\,\text{km/h}$), highest sustained PM2.5 baseline.
2. **Summer / Pre-Monsoon (Mar 1 – May 31)**: High surface temperatures ($35–45^\circ\text{C}$), intense thermal convection, deep PBLH ($>1800\,\text{m}$), moderate PM2.5 with occasional dust storm spikes.
3. **Monsoon (Jun 1 – Sep 30)**: Heavy precipitation ($>80\%$ of annual rainfall), strong wet aerosol deposition/washout, lowest annual PM2.5 ($<50\,\mu\text{g/m}^3$).
4. **Post-Monsoon (Oct 1 – Nov 30)**: Agricultural stubble burning window (`is_stubble_season = 1`), northwest transport wind alignment, rapid drop in PBLH, transition to winter stagnation.

---

## 11. Extreme-Event Requirements

Generating realistic severe and hazardous pollution episodes ($\text{PM}_{2.5} \ge 250\,\mu\text{g/m}^3$) requires strict joint environmental coherence:

```
                               COHERENT EXTREME-EVENT SYNTHESIS PROFILE
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Primary Drivers (Co-occurring):                                                                  │
│   1. Season Context: Post-Monsoon (Oct–Nov) OR Mid-Winter (Dec–Jan).                            │
│   2. Ventilation Deficit: Ventilation Index ≤ 2,500 m²/s (Wind Speed ≤ 10 km/h, PBLH ≤ 700 m).   │
│   3. Inversion Layer: Nocturnal minimum PBLH ≤ 350 m.                                            │
│   4. Upwind Emission Forcing: If Post-Monsoon, active fire counts ≥ 200/day in NW quadrant.      │
│   5. Precipitation Absence: Rainfall = 0.0 mm (zero washout).                                    │
│   6. Multi-Day Inertia: Prior day PM2.5 ≥ 180 µg/m³ (accumulation over 48–72 hours).             │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Critical Rule**: A synthetic record with $\text{PM}_{2.5} = 380\,\mu\text{g/m}^3$ generated alongside $\text{Rainfall} = 50\,\text{mm}$ or $\text{Ventilation Index} = 15,000\,\text{m}^2/\text{s}$ is **PHYSICALLY IMPOSSIBLE** and must be rejected by the validation filter.

---

## 12. Candidate Synthetic-Generation Architectures

To identify the optimal generator for Phase 7, 7 architectural families were systematically evaluated:

| Architecture Family | Temporal Preservation | Physics Constraint Enforceability | Extreme-Event Coherence | Leakage Safety | Computational Complexity | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Parametric Statistical Simulation** | Moderate | High (Exact bounds) | Low (Fails on heavy tails) | High | Minimal ($O(N)$) | Secondary baseline |
| **B. Conditional Resampling (Block Bootstrap)** | High | Very High (Empirical) | Moderate (Limited to seen events) | High | Low ($O(N)$) | Component of Hybrid |
| **C. Copula-Based Generation (Vine/Gaussian)** | Low–Mod | Moderate | Moderate (Tail copulas) | High | Moderate ($O(N D^2)$) | Statistical comparator |
| **D. Markov State-Transition (Regime-Switching)** | High | High (Regime-aware) | High (Multi-day chains) | High | Moderate ($O(N K^2)$) | Component of Hybrid |
| **E. Gaussian Process / State-Space Models** | High | Low (Difficult bounds) | Low (Gaussian assumption) | Moderate | High ($O(N^3)$) | Rejected |
| **F. Deep Conditional Generative ML (TimeGAN/VAE)** | High | Low (Black-box hallucinations) | High | Moderate (Risk of overfitting) | Very High (GPU required) | Deferred to Phase 9 |
| **G. Hybrid Physics-Informed Stochastic Generator** | **Very High** | **Very High (Exact rule engine)** | **Very High (Coupled priors)** | **Very High** | **Moderate ($O(N)$)** | **RECOMMENDED PRIMARY** |

---

## 13. Recommended Architecture: Hybrid Physics-Informed Stochastic Trajectory Generator (HP-STG)

The primary recommended architecture for Phase 7B is the **Hybrid Physics-Informed Stochastic Trajectory Generator (HP-STG)**:

```mermaid
graph TD
    A[1. Regime-Switching Markov Chain] -->|Sample Seasonal & Regime State Trajectory| B[2. Boundary Meteorological Generator]
    B -->|Sample Wind, Temp, Humidity, Rain, PBLH| C[3. Emission & Satellite Forcing Model]
    C -->|Sample Active Fires conditioned on Stubble Season & NW Wind| D[4. Atmospheric Mass-Balance & Dispersion ODE]
    D -->|Simulate Base Daily PM2.5 Mass Conc| E[5. Physics & Boundary Constraint Filter]
    E -->|Pass Hard Constraints: PM25>=0, VI=ws*pblh, Bounds| F[6. Deterministic Lag & Rolling Feature Engine]
    F -->|Output 35-Feature Prediction-Safe Observation| G[7. Validated Synthetic Trajectory Corpus]
```

### Generator Stages:
1. **Regime & Seasonal State Sampler**: A 4-state Markov transition model conditioned on month of year generating coherent multi-day weather regime sequences.
2. **Meteorological Vector Sampler**: Generates surface weather ($T, RH, \text{ws}, \text{PBLH}, \text{Rain}$) preserving joint empirical covariance per regime.
3. **Emission Forcing Engine**: Generates satellite active fire counts conditioned on `is_stubble_season` and transport vectors.
4. **Mass-Balance Stochastic ODE**:
   $$\frac{dC}{dt} = \frac{E(t)}{\text{PBLH}(t)} - k_{\text{disp}}(u, v) \cdot C(t) - k_{\text{washout}}(\text{Rain}) \cdot C(t) + \sigma_{\text{regime}} dW_t$$
5. **Deterministic Feature Derivation**: Computes all 24 lagged and rolling features directly from the simulated trajectory, ensuring 100% mathematical consistency.

---

## 14. Fallback Architecture: Regime-Stratified Block-Bootstrap with Copula Perturbation (RS-BBCP)

If the dynamical ODE in HP-STG exhibits calibration difficulties during Phase 7B:
- **Fallback Design**: Extract historical 14-day temporal blocks stratified by season and regime.
- Apply a calibrated Gaussian/t-Copula perturbation within the historical regime manifold to create novel environmental combinations while strictly preserving observed empirical physics.

---

## 15. Data Partitioning Strategy

To prevent scientific contamination and false performance claims, strict temporal data partitioning is mandated:

```
                                  CHRONOLOGICAL PARTITIONING TIMELINE
┌──────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│ HISTORICAL TRAINING / SYNTHESIS SOURCE       │ LOCKED REAL EVALUATION PERIOD (UNTOUCHED)        │
│ 2020-01-01 to 2021-12-31 (N = 731 Days)      │ 2022-01-01 to 2024-12-31 (N = 1,096 Days)        │
├──────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ • Permitted source for fitting generators    │ • STRICTLY FORBIDDEN FROM SYNTHETIC GENERATION   │
│ • Permitted for copula covariance estimation │ • ZERO ACCESS for distribution fitting           │
│ • Permitted for Markov transition matrices   │ • ZERO ACCESS for feature normalization          │
│ • Permitted for physical prior calibrations  │ • PRESERVED AS INDEPENDENT REAL BENCHMARK        │
└──────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

---

## 16. Leakage Prevention Strategy

Synthetic data pipelines are vulnerable to subtle forms of temporal leakage. Phase 7 enforces 6 structural isolation rules:

1. **Zero Future Lookahead**: Generator parameters, transition probabilities, and empirical quantiles must be fitted exclusively on historical data $\le 2021-12-31$.
2. **Isolated Test Set**: The locked 2022–2024 dataset must never be seen by the synthetic generator or used in tuning generator hyperparameters.
3. **Trajectory Directionality**: Synthetic time series must be generated forward in time ($t \to t+1$); rolling features must only look backward ($t-1, t-2, \dots$).
4. **Deterministic Feature Construction**: Lags and rolling features must be computed after generating raw state trajectories—never sampled independently.
5. **No Label Leakage**: Target $\text{PM}_{2.5}$ must not be used to condition input features that would be unknown at operational forecast time $t-1$.
6. **Automated Leakage Audit Script**: Phase 7 must implement `ml/src/modeling/phase7/leakage_audit.py` verifying 0 lookahead violations across generated corpora.

---

## 17. Physics Constraint Specification

Every generated synthetic batch must pass 10 automated physical validation tests:

1. **Non-Negativity**: $\forall t, \text{PM}_{2.5}(t) \ge 0.0\,\mu\text{g/m}^3, \text{PBLH}(t) \ge 150.0\,\text{m}, \text{Rainfall}(t) \ge 0.0\,\text{mm}, \text{WindSpeed}(t) \ge 0.0\,\text{km/h}$.
2. **Atmospheric Dispersion Consistency**: $\text{ventilation\_index\_1d} = \text{wind\_speed\_kmh\_lag\_1d} \times \frac{1000}{3600} \times \text{pblh\_1d} \pm 5\%$.
3. **Precipitation Logic**: $\text{rain\_event\_1d} = 1 \iff \text{rainfall\_1d} \ge 1.0\,\text{mm}$.
4. **Mathematical Rolling Identity**: $\text{pm25\_roll\_min\_7d} \le \text{pm25\_roll\_mean\_7d} \le \text{pm25\_roll\_max\_7d}$.
5. **Wind Vector Magnitude**: $|\sqrt{u^2 + v^2} - \text{wind\_speed}_{\text{m/s}}| \le 0.5\,\text{m/s}$.
6. **Relative Humidity Bound**: $0.0\% \le \text{humidity\_pct} \le 100.0\%$.
7. **Seasonal Stubble Boundary**: `is_stubble_season` $= 1$ strictly between Day-of-Year 288 (Oct 15) and 334 (Nov 30).
8. **Temperature Inversion Constraint**: In Winter, $\text{pblh\_min\_1d} \le 800.0\,\text{m}$.
9. **Washout Aerosol Upper Bound**: Active severe rain ($>30\,\text{mm}$) cannot co-occur with daily PM2.5 spike $>300\,\mu\text{g/m}^3$.
10. **Numerical Finiteness**: Zero NaN, Null, or Infinite values across all 35 features.

---

## 18. Statistical Validation Specification

To confirm that synthetic data accurately reflects observational distributions without mode collapse, the following metrics will be computed between real historical data and synthetic cohorts:

1. **Wasserstein-1 Distance ($W_1$)**: Computed across all 35 marginal feature distributions. Target: $W_1(\text{normalized}) \le 0.15$.
2. **Kolmogorov-Smirnov (KS) Test**: Two-sample KS test per feature. Target: Average KS statistic $D \le 0.12$.
3. **Jensen-Shannon Divergence (JSD)**: Binned probability density divergence. Target: $\text{JSD} \le 0.08$.
4. **Maximum Mean Discrepancy (MMD)**: Multi-dimensional Gaussian RBF kernel MMD on normalized feature space. Target: $\text{MMD}^2 \le 0.025$.
5. **Tail Quantile Error (Q95/Q99)**: Deviation in 95th and 99th percentiles. Target: $|\Delta Q_{95}| \le 15.0\,\mu\text{g/m}^3$.

---

## 19. Temporal Validation Specification

Temporal structure will be validated using:

1. **Autocorrelation Function (ACF) Distance**:
   $$\text{ACF Error} = \frac{1}{K} \sum_{k=1}^7 |\text{ACF}_{\text{real}}(k) - \text{ACF}_{\text{synth}}(k)| \le 0.08$$
2. **Rolling Volatility Preservation**: Comparison of $\sigma(\text{PM}_{2.5, \text{7d}})$ distributions (KS statistic $\le 0.10$).
3. **Episode Persistence Duration**: Average duration of consecutive days above $250\,\mu\text{g/m}^3$ ($4.2 \pm 2.1$ days real vs $4.0 \pm 2.3$ days synth).

---

## 20. Regime Validation Specification

Regime-specific fidelity will be evaluated across all 4 pollution regimes:

1. **Regime Proportion Preservation**:
   - Low: $31.3\% \pm 5\%$
   - Moderate: $27.4\% \pm 5\%$
   - High: $24.5\% \pm 5\%$
   - Extreme: $16.8\% \pm 5\%$
2. **Regime-Conditioned Residual Variance**:
   - Low regime $\sigma \in [7.5, 12.0]\,\mu\text{g/m}^3$
   - Extreme regime $\sigma \in [38.0, 52.0]\,\mu\text{g/m}^3$

---

## 21. Extreme-Event Validation Specification

Synthetic extreme events ($\text{PM}_{2.5} \ge 250\,\mu\text{g/m}^3$) will undergo joint-distribution auditing:
- $\ge 90\%$ of synthetic extreme events must occur during Post-Monsoon or Winter.
- $\ge 95\%$ of synthetic extreme events must exhibit $\text{Ventilation Index} \le 3,500\,\text{m}^2/\text{s}$.
- $0\%$ of synthetic extreme events may co-occur with rainfall $> 10\,\text{mm}$.

---

## 22. Acceptance Criteria (Go / No-Go Gate)

Synthetic data generated in Phase 7B will be accepted for Phase 8 only if all primary criteria are met:

```
                               PHASE 7 ACCEPTANCE CRITERIA MATRIX
┌───────────────────────────────────┬──────────────────────┬─────────────┬───────────┐
│ Metric / Test                     │ Target Threshold     │ Importance  │ Status    │
├───────────────────────────────────┼──────────────────────┼─────────────┼───────────┤
│ Physical Constraint Pass Rate     │ Exactly 100.0%       │ CRITICAL    │ Mandatory │
│ Temporal Leakage Violations       │ Exactly 0            │ CRITICAL    │ Mandatory │
│ Average Wasserstein-1 Distance    │ ≤ 0.15 (normalized)  │ PRIMARY     │ Required  │
│ Correlation Matrix Frobenius Dist │ ≤ 0.20               │ PRIMARY     │ Required  │
│ ACF Lag-1 to Lag-7 Mean Error     │ ≤ 0.08               │ PRIMARY     │ Required  │
│ Extreme Event Environmental Coher │ ≥ 95.0% Pass         │ PRIMARY     │ Required  │
│ Kolmogorov-Smirnov Stat Mean      │ ≤ 0.12               │ SECONDARY   │ Monitored │
│ Downstream Retraining Parity      │ ΔMAE ≤ +1.0 µg/m³    │ SECONDARY   │ Monitored │
└───────────────────────────────────┴──────────────────────┴─────────────┴───────────┘
```

---

## 23. Provenance Strategy

Every synthetic dataset produced in Phase 7 must be packaged with immutable provenance metadata:
1. **Directory**: `ml/data/synthetic/v1/`
2. **Files**:
   - `synthetic_dataset.csv`
   - `provenance_manifest.json`: Source dataset SHA-256 (`78b329fbc6...`), generator architecture (`HP-STG v1.0.0`), random seed (`42`), generation timestamp, sample count, parameter dictionary.
   - `checksums.txt`: SHA-256 hash of the generated synthetic CSV.
3. **Strict Versioning**: Independent semantic versions (`v1.0.0-synthetic`, `v2.0.0-synthetic`). Never overwrite existing synthetic files.

---

## 24. Reproducibility Strategy

The synthetic data generation engine must be 100% deterministic:
- Seeded pseudo-random number generation (`numpy.random.RandomState(seed)`).
- Double-generation audit:
  $$\text{Run A}(\text{seed}=42) - \text{Run B}(\text{seed}=42) \equiv 0.00\text{e}+00$$
- Automated test in `ml/tests/test_phase7.py` validating $\Delta \le 10^{-12}$.

---

## 25. Visualization Plan

Phase 7 will produce 14 publication-quality comparison figures under `ml/experiments/phase7/plots/`:

1. `1_real_vs_synthetic_pm25_distribution.png`: Overlaid KDE densities and quantile plots.
2. `2_feature_marginal_distributions.png`: 35-panel violin comparisons (Real vs Synthetic).
3. `3_correlation_matrix_comparison.png`: Side-by-side $35 \times 35$ feature correlation heatmaps and delta difference matrix.
4. `4_autocorrelation_acf_comparison.png`: ACF curves for Lags 1–30.
5. `5_seasonal_pm25_distributions.png`: Box plots across Winter, Summer, Monsoon, Post-Monsoon.
6. `6_regime_transition_probabilities.png`: Markov state transition matrices (Real vs Synthetic).
7. `7_extreme_event_environmental_scatter.png`: PM2.5 vs Ventilation Index colored by regime.
8. `8_pm25_vs_pblh_dispersion.png`: Boundary layer mixing height co-dependence.
9. `9_biomass_fire_response_curve.png`: Stubble fire count vs PM2.5 during post-monsoon.
10. `10_precipitation_washout_dynamics.png`: Rainfall events vs PM2.5 drop.
11. `11_synthetic_trajectory_timeseries.png`: Continuous 90-day synthetic sequence examples.
12. `12_wasserstein_distance_ranking.png`: Bar chart of per-feature Wasserstein distances.
13. `13_qq_plots_heavy_tails.png`: Quantile-Quantile plot comparing extreme upper tail.
14. `14_physical_constraint_audit_summary.png`: Zero-violation compliance dashboard.

---

## 26. Phase 7B–7E Implementation Roadmap

```
PHASE 7A: Technical Specification & Blueprint (CURRENT - ACCEPTED)
   ↓
PHASE 7B: Core Synthetic Engine Implementation (HP-STG Generator & Constraint Engine)
   ↓
PHASE 7C: Multi-Layer Validation & Extreme Episode Calibration
   ↓
PHASE 7D: Leakage Audit, Physical Boundary Audit & Reproducibility Verification
   ↓
PHASE 7E: Synthetic Data Packaging, Production Versioning (v1.0.0-synthetic) & Phase 8 Handover
```

---

## 27. Phase 8 Integration Contract

Synthetic data produced by Phase 7 will interface with Phase 8 under the following contract:

1. **Artifact Location**: `ml/data/synthetic/v1/synthetic_dataset_frozen.csv`.
2. **Schema Uniformity**: Exact match with `ml/models/production/v3/feature_registry.csv` (35 features + `pm25` + `date`).
3. **Training Corpus Expansion**: Phase 8 may augment historical training sets ($2020–2021$) with validated synthetic trajectories.
4. **Evaluation Benchmark**: The locked real test set ($2022–2024, N=1,096$) must remain 100% real observational data.
5. **Baseline Comparison**: Any model trained on synthetic data in Phase 8 must be compared directly against the frozen Phase 6 baseline (`MODEL_V3_PRODUCTION`, $R^2 = 0.9497, \text{MAE} = 17.02\,\mu\text{g/m}^3$).

---

## 28. Risks and Failure Modes

| Risk Description | Severity | Mitigation Strategy |
| :--- | :---: | :--- |
| **Mode Collapse** (under-representing rare episodes) | High | Regime-conditioned stratified sampling ensuring 15–20% representation of extreme events. |
| **Physical Incoherence** (severe smog during heavy rain) | Critical | Automated 10-point hard constraint rejection filter. |
| **Temporal Discontinuity** (jump artifacts between days) | High | Generate sequential trajectories (blocks $\ge 14$ days) with continuous ODE integration. |
| **Test Set Contamination** (indirect leakage) | Critical | Strict chronological source isolation ($\le 2021-12-31$ only) with cryptographic verification. |
| **Downstream Degradation** (synthetic data hurts model) | Moderate | Downstream utility parity test: model trained on Real+Synth must achieve $\Delta \text{MAE} \le 0.0$ on validation set. |

---

## 29. Scientific Limitations

The following scientific caveats must be formally acknowledged in all downstream reporting:

> **`SYNTHETIC DATA != OBSERVED DATA`**  
> Synthetic observations are stochastic realizations of an idealized physical-statistical model. They do not constitute empirical ground truth.

> **`PHYSICS-INFORMED != PHYSICALLY EXACT`**  
> Simplified mass-balance and atmospheric dispersion equations approximate boundary layer dynamics but do not resolve 3D turbulent fluid dynamics, photochemistry, or secondary aerosol formation mechanisms.

> **`STATISTICAL CONSISTENCY != CAUSAL VALIDITY`**  
> Replicating observational correlations between fire counts, ventilation, and PM2.5 does not establish causal emission attribution.

---

## 30. Final Go / No-Go Criteria & Decision

All Phase 7A design prerequisites have been rigorously established:
- [x] Complete repository audit performed.
- [x] All 5 core upstream SHA-256 hashes verified (`PASS`).
- [x] Feature semantics and bounds defined for all 35 prediction-safe features.
- [x] Physical relationship matrix and hard constraints formulated.
- [x] Temporal trajectory structure and multi-day autocorrelation analyzed.
- [x] Pollution-regime and seasonal stratification designed.
- [x] Hybrid Physics-Informed Stochastic Trajectory Generator (HP-STG) architecture selected and justified.
- [x] Strict chronological data partitioning and leakage prevention rules defined.
- [x] Multi-layer statistical, temporal, and physical validation framework established.
- [x] Quantitative acceptance criteria and failure modes documented.
- [x] Phase 6 production models and uncertainty layers preserved 100% unmodified.

---

### Formal Status Banner

```
============================================================
AtmosIQ Phase 7A
Physics-Informed Synthetic Data Design
============================================================

Repository audit:               PASS
Upstream lineage hashes:        PASS
Production model integrity:    PASS (Unmodified)
Production uncertainty:         PASS (Unmodified)
Feature registry (35 features): PASS (Unmodified)

Feature inventory:              COMPLETE
Physics relationship matrix:    COMPLETE
Temporal dependency analysis:   COMPLETE
Regime & seasonal design:       COMPLETE
Architecture selection:         COMPLETE (HP-STG Primary)
Partitioning & leakage rules:   COMPLETE
Validation framework:           COMPLETE
Acceptance criteria:            COMPLETE
Phase 7B–7E roadmap:            COMPLETE

============================================================
PHASE_7A_STATUS: COMPLETE
PHASE_7B_IMPLEMENTATION_READY: YES
============================================================
```
