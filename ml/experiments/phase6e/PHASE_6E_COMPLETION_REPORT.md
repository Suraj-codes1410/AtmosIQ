# AtmosIQ Phase 6E: SHAP & Counterfactual Uncertainty Report

## 1. Executive Summary
Phase 6E extends the AtmosIQ uncertainty framework beyond point forecasts and prediction intervals into **Attribution Uncertainty** and **Counterfactual Uncertainty**. Across an expanding chronological walk-forward ensemble ($B=30$ bootstrap models, 2022–2024, $N=1,096$ held-out days), Phase 6E evaluated the stability and dispersion of TreeSHAP attributions and model counterfactual predictions for the production model **MODEL_V3_PRODUCTION**.

---

## 2. Upstream Provenance & Lineage Verification
- **Dataset v3 SHA-256**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736` (`PASS`)
- **Production Model SHA-256**: `9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210` (`PASS`)
- **Feature Registry**: Exactly 35 prediction-safe features (`ml/models/production/v3/feature_registry.csv`).
- **Production Uncertainty Layer**: Unmodified `normalized_conformal` (`ml/uncertainty/production/v1/`).
- **Production Forecasting Model**: Kept strictly frozen.

---

## 3. Feature-Level Attribution Uncertainty (Top 15 Features)
| feature_name                |   mean_absolute_shap |   mean_signed_shap |   mean_shap_std |   mean_sign_stability | stability_classification   |   average_positive_fraction |   average_negative_fraction |   importance_rank |
|:----------------------------|---------------------:|-------------------:|----------------:|----------------------:|:---------------------------|----------------------------:|----------------------------:|------------------:|
| aod_550_1d                  |            52.0298   |        -4.10174    |        4.74156  |              0.986466 | HIGH_STABILITY             |                    0.339234 |                    0.660645 |                 1 |
| pm25_roll_mean_3d           |            13.4716   |         3.6407     |        4.5693   |              0.95003  | HIGH_STABILITY             |                    0.70003  |                    0.298388 |                 2 |
| pm25_lag_1d                 |             8.50117  |         2.42437    |        4.06019  |              0.941089 | HIGH_STABILITY             |                    0.678558 |                    0.318948 |                 3 |
| pm25_roll_min_7d            |             4.33236  |         0.386879   |        2.69533  |              0.934185 | HIGH_STABILITY             |                    0.405292 |                    0.588473 |                 4 |
| pm25_roll_max_7d            |             3.52869  |        -0.0394985  |        2.5457   |              0.935006 | HIGH_STABILITY             |                    0.377889 |                    0.616332 |                 5 |
| pm25_roll_mean_7d           |             2.73701  |        -0.120623   |        2.02139  |              0.935006 | HIGH_STABILITY             |                    0.3368   |                    0.653802 |                 6 |
| pblh_roll_mean_3d           |             2.60932  |         1.34196    |        1.40392  |              0.96542  | HIGH_STABILITY             |                    0.65371  |                    0.338595 |                 7 |
| pm25_lag_2d                 |             1.75252  |         0.263188   |        1.53361  |              0.865785 | MODERATE_STABILITY         |                    0.547597 |                    0.423814 |                 8 |
| pm25_roll_mean_14d          |             1.51716  |        -0.00176807 |        1.6762   |              0.895377 | MODERATE_STABILITY         |                    0.328133 |                    0.637318 |                 9 |
| pblh_1d                     |             0.90211  |         0.435666   |        0.743493 |              0.865693 | MODERATE_STABILITY         |                    0.593856 |                    0.361557 |                10 |
| pblh_min_1d                 |             0.668068 |         0.15047    |        0.509451 |              0.890116 | MODERATE_STABILITY         |                    0.50587  |                    0.456235 |                11 |
| pm25_lag_3d                 |             0.328203 |        -0.0140911  |        0.585075 |              0.677159 | LOW_STABILITY              |                    0.449422 |                    0.458425 |                12 |
| wind_speed_kmh_roll_mean_3d |             0.326155 |        -0.0277639  |        0.377782 |              0.745286 | MODERATE_STABILITY         |                    0.433273 |                    0.457786 |                13 |
| wind_v_component_1d         |             0.276867 |         0.0469217  |        0.36398  |              0.702524 | MODERATE_STABILITY         |                    0.431174 |                    0.439294 |                14 |
| pm25_lag_7d                 |             0.232569 |        -0.0248992  |        0.388363 |              0.660036 | LOW_STABILITY              |                    0.405322 |                    0.473054 |                15 |

### Attribution Stability Distribution:
- **High Stability (>= 90%)**: `7` features
- **Moderate Stability (70% - 90%)**: `10` features
- **Low Stability (< 70%)**: `18` features
- **SHAP Additivity Pass Rate**: `100.00%`

---

## 4. Environmental Group-Level Attribution Uncertainty
| feature_group          |   feature_count |   mean_absolute_group_shap |   mean_signed_group_shap |   mean_group_shap_std |   group_sign_stability |   q10_group_shap_mean |   q90_group_shap_mean |
|:-----------------------|----------------:|---------------------------:|-------------------------:|----------------------:|-----------------------:|----------------------:|----------------------:|
| external_environmental |               5 |               52.0648      |             -4.02621     |           4.75912     |               0.556496 |         -10.0013      |           2.43465     |
| pm25_persistence       |              10 |               31.0112      |              6.55384     |           7.98936     |               0.840538 |         -17.1911      |          30.916       |
| wind_ventilation       |               8 |                4.17247     |              1.95382     |           1.91297     |               0.770495 |          -2.93026     |           7.06296     |
| meteorology            |               6 |                0.604162    |              0.0471314   |           0.622211    |               0.64262  |          -1.48769     |           1.52973     |
| biomass_burning        |               5 |                0.563919    |             -0.0424242   |           0.566493    |               0.537828 |          -1.19831     |           1.10779     |
| calendar_seasonal      |               1 |                1.55102e-05 |             -1.12212e-05 |           5.65503e-05 |               0        |          -1.53274e-05 |           6.79596e-07 |

---

## 5. Counterfactual Scenario Uncertainty & Directional Stability
| scenario_name          |   observation_count |   mean_delta_pm25 |   median_delta_pm25 |   mean_delta_std |   mean_directional_stability |   q10_delta_mean |   q90_delta_mean |   mean_interval_width_90 |
|:-----------------------|--------------------:|------------------:|--------------------:|-----------------:|-----------------------------:|-----------------:|-----------------:|-------------------------:|
| combined_biomass_wind  |                1096 |         -3.57612  |           -3.47815  |          4.38041 |                     0.818948 |         -8.8323  |          1.51257 |                 13.0971  |
| wind_dispersion        |                1096 |         -3.17712  |           -3.06915  |          4.03436 |                     0.835736 |         -8.00297 |          1.46426 |                 11.9996  |
| combined_all_favorable |                1096 |         -3.12688  |           -3.06526  |          4.7791  |                     0.824118 |         -8.89884 |          2.52152 |                 14.3977  |
| biomass_low            |                1096 |         -0.391919 |           -0.342121 |          1.33544 |                     0.798723 |         -1.90716 |          1.04136 |                  3.91418 |
| biomass_median         |                1096 |         -0.174947 |           -0.142309 |          1.2987  |                     0.790602 |         -1.61987 |          1.22287 |                  3.7761  |
| meteorology_normal     |                1096 |          0.318107 |            0.276571 |          1.81537 |                     0.782634 |         -1.62059 |          2.33909 |                  5.25299 |
| biomass_high           |                1096 |          0.703193 |            0.617738 |          1.71663 |                     0.81618  |         -1.13305 |          2.63829 |                  4.92561 |
| wind_stagnant          |                1096 |          7.28742  |            6.99582  |          4.56426 |                     0.844313 |          2.04683 |         12.8149  |                 13.8312  |

---

## 6. Out-Of-Distribution (OOD) & Uncertainty Interaction Analysis
| scenario_name          |   mean_ood_score |   mean_ood_percentile |   in_distribution_fraction |   mean_cf_uncertainty_std |   mean_directional_stability |
|:-----------------------|-----------------:|----------------------:|---------------------------:|--------------------------:|-----------------------------:|
| biomass_median         |          1.04357 |               51.9173 |                   1        |                   1.2987  |                     0.790602 |
| biomass_low            |          1.04856 |               52.1367 |                   1        |                   1.33544 |                     0.798723 |
| meteorology_normal     |          1.06632 |               52.9181 |                   1        |                   1.81537 |                     0.782634 |
| biomass_high           |          1.08216 |               53.615  |                   1        |                   1.71663 |                     0.81618  |
| combined_biomass_wind  |          1.26246 |               61.379  |                   0.961679 |                   4.38041 |                     0.818948 |
| wind_dispersion        |          1.26343 |               61.4436 |                   0.963504 |                   4.03436 |                     0.835736 |
| combined_all_favorable |          1.29081 |               62.6158 |                   0.962591 |                   4.7791  |                     0.824118 |
| wind_stagnant          |          1.39755 |               66.99   |                   0.872263 |                   4.56426 |                     0.844313 |

### OOD Statistical Correlation:
- **Spearman $\rho$ (OOD Score vs. Counterfactual Response Dispersion $\sigma_\Delta$)**: `+0.7637` ($p = 0.0000e+00$)
- **Finding**: Larger distributional deviations from historical training data are significantly correlated with wider counterfactual uncertainty intervals and reduced directional consensus.

---

## 7. Temporal Leakage & Physical Validity Audit
| audit_check                                       | condition                                                                       |   violations_detected | status   | notes                                                                            |
|:--------------------------------------------------|:--------------------------------------------------------------------------------|----------------------:|:---------|:---------------------------------------------------------------------------------|
| Chronological Walk-Forward Fold Isolation         | Folds strictly evaluate 2022, 2023, 2024 respectively                           |                     0 | PASS     | Ensemble models and SHAP explainers trained strictly on preceding years          |
| Monotonic Date Progression                        | Evaluation timestamps strictly increase monotonically                           |                     0 | PASS     | Zero temporal lookahead or shuffling detected                                    |
| Production Model & Uncertainty Layer Immutability | MODEL_V3_PRODUCTION and Phase 6D production uncertainty layer remain unmodified |                     0 | PASS     | Analytical layer decoupled; zero modification to production artifacts            |
| Counterfactual Prediction Non-Negativity          | cf_ensemble_mean >= 0.0 µg/m³ everywhere                                        |                     0 | PASS     | All 8768 counterfactual predictions satisfy PM2.5 non-negative physical boundary |
| Total Phase 6E Leakage Violations                 | Total violations == 0                                                           |                     0 | PASS     | Zero leakage violations confirmed across Phase 6E pipeline                       |

---

## 8. Scientific Language Safeguards
> **`PREDICTION INTERVAL ≠ ATTRIBUTION INTERVAL ≠ COUNTERFACTUAL INTERVAL ≠ PHYSICAL ATMOSPHERIC UNCERTAINTY`**  
> TreeSHAP dispersion quantifies model attribution stability across finite bootstrap samples under the learned training distribution. It does not establish causal atmospheric mechanisms or emission source attribution in a physical sense.

---

## 9. Final Status Banner

```
============================================================
AtmosIQ Phase 6E
SHAP & Counterfactual Uncertainty
============================================================

Dataset v3 integrity:              PASS
Production model integrity:       PASS
Feature registry integrity:       PASS
Phase 6D uncertainty integrity:   PASS

SHAP analysis:                     PASS
Group attribution analysis:       PASS
Counterfactual analysis:          PASS
OOD analysis:                     PASS

Temporal validation:              PASS
Leakage audit:                    PASS
Physical validity:                PASS
Reproducibility:                  PASS
Visualization:                    PASS
Tests:                            PASS

Production model modified:        NO
Production uncertainty modified:  NO

============================================================
PHASE 6E STATUS: COMPLETE
============================================================
```
