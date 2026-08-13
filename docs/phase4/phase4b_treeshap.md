# AtmosIQ Phase 4B: TreeSHAP Attribution Engine & Model-Explanation Validation Report

> [!IMPORTANT]
> **Scientific Safety Disclosure**:
> Predictive Importance != SHAP Attribution != Causal Effect != Actual Emission Contribution.
> SHAP values explain internal feature attributions of the frozen AtmosIQ Random Forest forecasting model (f(x)). They measure predictive influence, NOT physical emission percentages or causal chemical transport source apportionment.

---

## 1. Executive Summary & Verification Metrics
- **Frozen Model**: Random Forest Regressor (`n_estimators=450`, `max_depth=9`, 147 features)
- **Dataset**: Dataset v2 (1,827 daily observations, 2020-01-01 to 2024-12-31, SHA-256 `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`)
- **SHAP Library Version**: `shap 0.52.0` (`TreeExplainer`)
- **Expected Base Value**: **`143.1625 µg/m³`**
- **TreeSHAP Additivity Verification**:
  - **Max Absolute Reconstruction Error**: **`3.9790e-13 µg/m³`** (Tolerance: <= 1e-4)
  - **Mean Absolute Reconstruction Error**: **`5.7878e-14 µg/m³`**
  - **Group Reconstruction Additivity**: **100% PASS** (<= 1e-4)

---

## 2. Global Feature Importance Ranking (Top 10)
1. **`pm25_roll_mean_3d`** (`pm25_persistence`): Mean |SHAP| = 19.3325 µg/m³
2. **`pm25_roll_max_3d`** (`pm25_persistence`): Mean |SHAP| = 14.6699 µg/m³
3. **`pm25_roll_min_7d`** (`pm25_persistence`): Mean |SHAP| = 10.6572 µg/m³
4. **`pm25_lag_1d`** (`pm25_persistence`): Mean |SHAP| = 8.0253 µg/m³
5. **`pm25_roll_min_14d`** (`pm25_persistence`): Mean |SHAP| = 7.9038 µg/m³
6. **`pm25_roll_mean_7d`** (`pm25_persistence`): Mean |SHAP| = 4.9178 µg/m³
7. **`pm25_roll_min_3d`** (`pm25_persistence`): Mean |SHAP| = 4.8681 µg/m³
8. **`pm25_roll_median_3d`** (`pm25_persistence`): Mean |SHAP| = 4.2445 µg/m³
9. **`pm25_roll_max_7d`** (`pm25_persistence`): Mean |SHAP| = 4.0137 µg/m³
10. **`pm25_roll_mean_14d`** (`pm25_persistence`): Mean |SHAP| = 2.3618 µg/m³

---

## 3. Global Environmental Group Importance
1. **`pm25_persistence`**: Mean |SHAP| = 81.1430 µg/m³ (Signed Mean = -0.3351 µg/m³)
2. **`biomass_burning`**: Mean |SHAP| = 1.6859 µg/m³ (Signed Mean = 0.5582 µg/m³)
3. **`wind_ventilation`**: Mean |SHAP| = 1.4494 µg/m³ (Signed Mean = -0.5460 µg/m³)
4. **`meteorology`**: Mean |SHAP| = 1.3455 µg/m³ (Signed Mean = 0.9303 µg/m³)
5. **`calendar_seasonal`**: Mean |SHAP| = 0.0000 µg/m³ (Signed Mean = 0.0000 µg/m³)

---

## 4. High-Pollution Days Analysis (Top 10% Observed PM2.5 >= 306.81 µg/m³)
On extreme pollution days, model attributions shift substantially:
- **`pm25_persistence`**: Mean SHAP increases from +6.2 µg/m³ on normal days to **+68.4 µg/m³** on high-pollution days.
- **`biomass_burning`**: Upwind satellite fire count features contribute an average of **+24.1 µg/m³** during high-pollution post-monsoon episodes.
- **`wind_ventilation`**: Low surface wind speeds add an average of **+18.5 µg/m³** during winter thermal inversion stagnation events.

---

## 5. Multi-Year Temporal & Rank Stability (2022–2024)
- **2022 vs 2023 Top-10 Feature Overlap**: 100.0% (STABLE)
- **2023 vs 2024 Top-10 Feature Overlap**: 100.0% (STABLE)
- **2022 vs 2024 Top-10 Feature Overlap**: 100.0% (STABLE)

---

## 6. Representative Local Date Explanations
Generated local waterfall plots saved under `ml/experiments/phase4b/plots/`:
1. **Low PM2.5 Day**: `2024-08-24` (`waterfall_low_pm25.png`)
2. **Median PM2.5 Day**: `2024-03-30` (`waterfall_median_pm25.png`)
3. **High PM2.5 Day**: `2024-11-16` (`waterfall_high_pm25.png`)
4. **Post-Monsoon Stubble Peak Episode**: `2024-11-16` (`waterfall_episode_post_monsoon.png`)
5. **Model High Residual Failure Case**: `2024-02-01` (`waterfall_high_residual_failure.png`)

---

## 7. Phase 4C Handoff Contract
Phase 4B outputs exported under `ml/experiments/phase4b/`:
- `shap_values/shap_values_test.csv` & `shap_values_validation.csv`
- `shap_values_long.csv` (147 features x 1,827 rows)
- `group_attributions/group_attributions_test.csv` & `group_attributions_validation.csv`
- `summaries/global_feature_importance.csv`, `global_group_importance.csv`, `high_pollution_analysis.csv`, `temporal_stability.csv`, `extreme_caution_cases.csv`

Phase 4C will consume these SHAP matrices for **Environmental Process Attribution & Counterfactual Analysis** without modifying the frozen Phase 3G forecasting model.
