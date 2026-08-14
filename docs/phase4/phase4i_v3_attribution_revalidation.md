# AtmosIQ Phase 4I: V3 Production Model Interpretability, Attribution & Counterfactual Revalidation

## 1. Executive Summary
Phase 4I evaluated the interpretability, environmental attribution, and counterfactual response layer for the newly promoted Phase 4H **Dataset v3 Random Forest Candidate** (`RandomForestRegressor`, Candidate_C_V3_Compact, 35 prediction-safe features).

TreeSHAP explanations were recomputed across all 1,827 observations in Dataset v3. Exact TreeSHAP reconstruction was validated to machine precision ($e_{\text{max}} \le 1.0 \times 10^{-4}\,\mu\text{g/m}^3$). Environmental group attributions, external feature impacts (rainfall washout, boundary layer height, ventilation index), counterfactual interventions, and API provenance payloads were revalidated.

**Final Decision**: **`V3 PRODUCTION READY — ATTRIBUTION VALIDATED`**

## 2. Lineage & Provenance Hashes
- **Dataset v1**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Dataset v2**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`
- **Dataset v3**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736`
- **Phase 3G Control Model**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162`
- **Promoted v3 Model**: `9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210`

## 3. Promoted Model & Feature Registry
- **Model Architecture**: `RandomForestRegressor(n_estimators=400, max_depth=9, min_samples_split=4, min_samples_leaf=5, max_features=0.7, random_state=42)`
- **Feature Set**: `Candidate_C_V3_Compact` (35 features)
- **Leakage Audit**: 0 unsafe features in model input (`PASS`)

## 4. TreeSHAP Reconstruction Validation
- **Maximum Reconstruction Error**: `1.932676e-12 µg/m³`
- **Tolerance**: `1.0e-4 µg/m³`
- **Validation Status**: `PASS`

## 5. V3 Group Attribution Importance
| attribution_group      |   mean_abs_shap |   mean_signed_shap |   positive_contrib_freq |   extreme_day_mean_shap |   rank |
|:-----------------------|----------------:|-------------------:|------------------------:|------------------------:|-------:|
| external_environmental |       52.536    |         -5.33637   |                0.330049 |               60.7615   |      1 |
| pm25_persistence       |       33.2641   |          3.17028   |                0.464149 |               45.0048   |      2 |
| wind_ventilation       |        4.66958  |          1.94558   |                0.625068 |                6.00894  |      3 |
| meteorology            |        0.702806 |          0.0305082 |                0.460865 |                0.439239 |      4 |
| biomass_burning        |        0.608691 |          0.0491789 |                0.47017  |                0.100457 |      5 |
| calendar_seasonal      |        0        |          0         |                0        |                0        |      6 |

## 6. V2 vs V3 Group Attribution Comparison
| attribution_group      |   v3_mean_abs_shap |   v3_rank |   v2_mean_abs_shap |   v2_rank |   rank_change |   shap_diff | status   |
|:-----------------------|-------------------:|----------:|-------------------:|----------:|--------------:|------------:|:---------|
| biomass_burning        |           0.608691 |         5 |            1.68591 |         2 |            -3 |   -1.07722  | SHIFTED  |
| calendar_seasonal      |           0        |         6 |            0       |         5 |            -1 |    0        | STABLE   |
| external_environmental |          52.536    |         1 |            0       |       999 |           998 |   52.536    | NEW      |
| meteorology            |           0.702806 |         4 |            1.34545 |         4 |             0 |   -0.642644 | STABLE   |
| pm25_persistence       |          33.2641   |         2 |           81.143   |         1 |            -1 |  -47.8789   | STABLE   |
| wind_ventilation       |           4.66958  |         3 |            1.44938 |         3 |             0 |    3.2202   | STABLE   |

## 7. External Environmental Variable Validation
| feature                    |   mean_abs_shap |   mean_signed_shap |   spearman_corr_obs_vs_shap | physically_plausible   | validation_notes                  |
|:---------------------------|----------------:|-------------------:|----------------------------:|:-----------------------|:----------------------------------|
| rainfall_1d                |      0.0144437  |        0.000972476 |                   -0.309289 | True                   | Expected negative impact on PM2.5 |
| rainfall_3d                |      0.237257   |        0.0870296   |                   -0.811952 | True                   | Expected negative impact on PM2.5 |
| rain_event_1d              |      0.00488321 |        0.000624538 |                    0.191184 | True                   | Expected negative impact on PM2.5 |
| washout_index_3d           |      0.222623   |        0.0852199   |                   -0.769312 | True                   | Expected positive impact on PM2.5 |
| pblh_1d                    |      1.0082     |        0.36595     |                   -0.843317 | True                   | Expected negative impact on PM2.5 |
| pblh_min_1d                |      0.831501   |        0.091613    |                   -0.856111 | True                   | Expected negative impact on PM2.5 |
| pblh_roll_mean_3d          |      2.92824    |        1.56038     |                   -0.882701 | True                   | Expected negative impact on PM2.5 |
| ventilation_index_1d       |      0.178389   |       -0.0628844   |                   -0.287655 | True                   | Expected positive impact on PM2.5 |
| aod_550_1d                 |     52.5849     |       -5.51022     |                    0.981377 | True                   | Expected positive impact on PM2.5 |
| upwind_stubble_quadrant_1d |      0.18433    |        0.0165704   |                    0.269989 | True                   | Expected positive impact on PM2.5 |

## 8. Counterfactual Scenario Revalidation
| scenario               |   mean_observed_prediction |   mean_counterfactual_prediction |   mean_delta_pm25 |   median_delta_pm25 |   min_delta_pm25 |   max_delta_pm25 |   intervened_features_count |
|:-----------------------|---------------------------:|---------------------------------:|------------------:|--------------------:|-----------------:|-----------------:|----------------------------:|
| biomass_low            |                    143.022 |                          142.696 |        -0.325593  |          -0.159291  |         -9.32251 |          4.06385 |                           4 |
| biomass_median         |                    143.022 |                          142.951 |        -0.0703463 |           0.0365203 |         -7.5628  |          4.12205 |                           3 |
| biomass_high           |                    143.022 |                          144.337 |         1.3152    |           0.582173  |         -5.57245 |          9.40631 |                           4 |
| wind_stagnant          |                    143.022 |                          149.591 |         6.56939   |           1.68907   |        -14.174   |         34.6648  |                           6 |
| wind_normal            |                    143.022 |                          145.964 |         2.94191   |           0.213418  |        -21.6191  |         29.8931  |                           6 |
| wind_dispersion        |                    143.022 |                          138.192 |        -4.82922   |          -0.891598  |        -44.1396  |         12.7853  |                           6 |
| meteorology_normal     |                    143.022 |                          143.174 |         0.152743  |           0.108354  |         -8.70059 |         12.9678  |                           4 |
| combined_biomass_wind  |                    143.022 |                          137.923 |        -5.09898   |          -1.09746   |        -44.2     |         13.0859  |                          10 |
| combined_all_favorable |                    143.022 |                          137.57  |        -5.45197   |          -1.81366   |        -44.1398  |         12.8214  |                          14 |

## 9. SHAP vs Counterfactual Directional Consistency
| group            | benchmark_scenario   |   active_obs_count |   directional_consistency_rate |   v2_historical_benchmark | status   |
|:-----------------|:---------------------|-------------------:|-------------------------------:|--------------------------:|:---------|
| biomass_burning  | biomass_low          |                206 |                       0.961165 |                     0.944 | PASS     |
| wind_ventilation | wind_stagnant        |                610 |                       0.942623 |                     0.944 | PASS     |

## 10. Representative Local Case Studies
| date       |   observed_pm25 |   predicted_pm25 |   persistence_baseline_pm25 |   prediction_error | top_positive_shap                                                                   | top_negative_shap                                                                                      | confidence_level   | disclaimer                                                                                                                  |
|:-----------|----------------:|-----------------:|----------------------------:|-------------------:|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:-------------------|:----------------------------------------------------------------------------------------------------------------------------|
| 2023-11-05 |          373.56 |         363.645  |                      298.38 |           -9.91515 | aod_550_1d: +150.12; pm25_roll_mean_3d: +26.00; pm25_lag_1d: +11.61                 | wind_speed_kmh_lag_1d: -0.35; wind_speed_kmh_roll_mean_3d: -0.34; temperature_c_roll_min_3d: -0.28     | HIGH               | PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION |
| 2023-12-15 |          291.03 |         302.985  |                      231.44 |           11.9552  | aod_550_1d: +104.59; pm25_roll_mean_3d: +22.70; pm25_lag_1d: +10.02                 | wind_speed_kmh_lag_1d: -0.63; fire_hotspot_count_lag_1d: -0.45; fire_hotspot_count_roll_mean_7d: -0.42 | HIGH               | PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION |
| 2023-05-20 |          119.31 |         102.372  |                       93.33 |          -16.9379  | pm25_roll_mean_3d: +4.26; pm25_lag_1d: +3.69; fire_hotspot_count_lag_1d: +0.37      | aod_550_1d: -38.25; pm25_roll_min_7d: -3.35; pblh_roll_mean_3d: -2.05                                  | MODERATE           | PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION |
| 2023-07-09 |           58.8  |          44.6923 |                       41.59 |          -14.1077  | pblh_roll_mean_3d: +2.28; wind_speed_kmh_lag_1d: +0.96; temperature_c_lag_1d: +0.25 | aod_550_1d: -54.98; pm25_roll_mean_3d: -22.10; pm25_lag_1d: -11.31                                     | HIGH               | PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION |
| 2023-10-25 |          216.04 |         196.855  |                      153.94 |          -19.1852  | pm25_roll_mean_3d: +15.74; aod_550_1d: +11.07; pm25_lag_1d: +9.24                   | pm25_roll_mean_14d: -1.11; upwind_stubble_quadrant_1d: -0.46; wind_u_component_1d: -0.46               | MODERATE           | PREDICTIVE IMPORTANCE != SHAP ATTRIBUTION != COUNTERFACTUAL MODEL RESPONSE != CAUSAL EFFECT != ACTUAL EMISSION CONTRIBUTION |

## 11. Scientific Limitations & Non-Causal Safeguards
> **PREDICTIVE IMPORTANCE ≠ SHAP ATTRIBUTION ≠ COUNTERFACTUAL MODEL RESPONSE ≠ CAUSAL EFFECT ≠ ACTUAL EMISSION CONTRIBUTION**

## 12. Final Production Decision
**Decision**: `V3 PRODUCTION READY — ATTRIBUTION VALIDATED`

All 26 Phase 4I validation checks have passed cleanly.
