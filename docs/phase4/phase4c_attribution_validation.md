# AtmosIQ Phase 4C: Environmental Attribution Validation & Event-Level Attribution Report

> [!IMPORTANT]
> **Scientific Safety Disclosure**:
> AtmosIQ Phase 4C validates the environmental plausibility and consistency of model-derived SHAP explanations against independent observational indicators. It does NOT establish causal emission-source contributions. SHAP values quantify the contribution of model features to the model prediction. Historical PM2.5 features contain integrated information from multiple physical sources and therefore cannot be interpreted as isolated emission-source contributions. Biomass-burning SHAP should be interpreted as "the contribution of biomass-burning-related predictors to the model prediction", NOT as "the percentage of PM2.5 caused by stubble burning."

---

## 1. Executive Summary & Core Answers
- **Frozen Model**: Random Forest Regressor (`n_estimators=450`, `max_depth=9`, SHA-256 `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162`)
- **Dataset**: Dataset v2 (1,827 daily observations, 2020-01-01 to 2024-12-31, SHA-256 `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`)
- **Overall Environmental Support Confidence**: **`43.5%`** of observations demonstrate Moderate or High environmental support (0.8% High, 42.7% Moderate).

### Answers to Core Scientific Questions:
1. **Q1 (Biomass Co-occurrence)**: **YES**. Biomass burning SHAP attributions strongly correlate with satellite MODIS/VIIRS fire counts (Spearman $r = 0.5291$, $p = 2.6106e-132$). On high-fire days ($\ge 75\text{th}$ percentile), mean biomass SHAP increases to **`3.04 µg/m³`**.
2. **Q2 (Wind/Ventilation Consistency)**: **YES**. Wind/ventilation SHAP attributions correlate negatively with surface wind speed (Spearman $r = -0.5587$, $p = 1.6394e-150$). Low wind speeds ($\le 5\text{ km/h}$) contribute an average of **`+2.57 µg/m³`** due to atmospheric stagnation.
3. **Q3 (Meteorological Plausibility)**: **YES**. Meteorological SHAP attributions correlate with cold winter temperatures ($r = -0.4762$) and high relative humidity ($r = 0.4467$), consistent with boundary layer thermal inversion dynamics and secondary aerosol hydro-swelling.
4. **Q4 (Regime Differences)**: **YES**. Group SHAP attributions change substantially across seasons (e.g. Post-Monsoon is dominated by `biomass_burning`, Winter is dominated by `wind_ventilation` stagnation, and Monsoon is dominated by rain washouts).
5. **Q5 (Multi-Year Stability)**: **YES**. Multi-year rank correlation across 2020–2024 demonstrates **100% Top-10 feature overlap** between 2023 and 2024 ($r = 1.0000$, $p = 0.0000$).
6. **Q6 (Event Attribution)**: **YES**. Detected **`110`** extreme pollution episodes ($\ge 90\text{th}$ percentile threshold $306.81\text{ µg/m³}$), each fully documented in `event_catalog.csv`.

---

## 2. Biomass Burning Validation
- **Spearman Rank Correlation**: **`0.5291`** ($p = 2.6106e-132$)
- **Mean Biomass SHAP (Low Fire $\le 25\text{th}$ percentile)**: `-0.43 µg/m³`
- **Mean Biomass SHAP (High Fire $\ge 75\text{th}$ percentile)**: **`3.04 µg/m³`**
- **$P(\text{High SHAP} \mid \text{High Fire})$**: **`65.7%`**

---

## 3. Wind & Dispersion Validation
- **Spearman Rank Correlation**: **`-0.5587`** ($p = 1.6394e-150$)
- **Mean Wind SHAP (Stagnation Regime $\le 5\text{ km/h}$)**: **`+2.57 µg/m³`**
- **Mean Wind SHAP (Dispersion Regime $\ge 12\text{ km/h}$)**: `-0.98 µg/m³`

---

## 4. Counter-Evidence & Conflict Detection
- **Identified Conflict Cases**: **`47`** observations flagged in `attribution_conflicts.csv`.
- These cases highlight model limitations (e.g. upwind fire activity occurring when local transport winds bypass Delhi, or satellite cloud cover masking fires).

---

## 5. Phase 4D Recommendations
Phase 4C outputs exported under `ml/experiments/phase4c/`:
- `attribution_validation_summary.csv`
- `biomass_validation.csv`, `wind_validation.csv`, `meteorology_validation.csv`
- `seasonal_validation.csv`, `temporal_validation.csv`
- `event_catalog.csv`, `event_attributions.csv`
- `attribution_conflicts.csv`, `confidence_scores.csv`, `statistical_tests.csv`

Proceed to **Phase 4D: Source Category Attribution & Counterfactual Simulation Engine**.
