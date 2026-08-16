# AtmosIQ Phase 6B: Ensemble-Based Predictive Uncertainty

## 1. Executive Summary
Phase 6B investigates whether ensemble and model variation provides a meaningful empirical signal of predictive uncertainty around the frozen **MODEL_V3_PRODUCTION** model. Across an expanding chronological walk-forward evaluation (2022–2024, N = 1,096 days), we constructed controlled Bootstrap Ensembles (B=30), Random-Seed Ensembles (N=30), and Model-Family Ensembles (N=4), and rigorously tested whether ensemble spread correlates with actual out-of-sample prediction error.

## 2. Immutable Lineage & Provenance
- **Dataset v1 SHA-256**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Dataset v2 SHA-256**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`
- **Dataset v3 SHA-256**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736`
- **Production Model SHA-256**: `9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210`
- **Feature Count**: Exactly 35 prediction-safe features.
- **Production Model Binary**: Preserved frozen in `ml/models/production/v3/model.joblib`.

## 3. Paradigm Comparison (Control vs. Bootstrap vs. Seed vs. Family)
| paradigm                  |   mae_ugm3 |   rmse_ugm3 |       r2 |   mean_spread_ugm3 |   spearman_spread_error_corr |   coverage_90pct |   mpiw_90pct_ugm3 |   winkler_score_90pct |   extreme_episode_coverage_90pct |
|:--------------------------|-----------:|------------:|---------:|-------------------:|-----------------------------:|-----------------:|------------------:|----------------------:|---------------------------------:|
| Frozen_Model_Control      |    17.0168 |     21.8817 | 0.949952 |           0        |                     0        |        0         |           0       |                 0     |                        0         |
| Bootstrap_Ensemble_B30    |    17.031  |     21.8606 | 0.950048 |           5.41229  |                     0.280451 |        0.254562  |          16.3611  |               219.036 |                        0.301508  |
| Random_Seed_Ensemble_N30  |    17.0204 |     21.8962 | 0.949886 |           0.902783 |                     0.311902 |        0.0374088 |           2.73566 |               316.328 |                        0.0326633 |
| Model_Family_Diversity_N4 |    16.915  |     21.8636 | 0.950034 |           4.18514  |                     0.196334 |        0.160584  |           9.36176 |               266.088 |                        0.198492  |

## 4. Spread vs. Actual Prediction Error Correlation
- **Spearman Rank Correlation (rho)**: `0.2805`
- **Quintile Error Breakdown**:
| ensemble_type   | spread_quintile   |   count |   mean_spread_ugm3 |   median_spread_ugm3 |   mae_ugm3 |   rmse_ugm3 |   medae_ugm3 |   empirical_coverage_90pct |
|:----------------|:------------------|--------:|-------------------:|---------------------:|-----------:|------------:|-------------:|---------------------------:|
| bootstrap       | Q1 (Lowest)       |     220 |            2.33543 |              2.37592 |    10.809  |     12.6136 |      11.034  |                   0.163636 |
| bootstrap       | Q2                |     219 |            3.60832 |              3.61297 |    13.7643 |     16.8809 |      12.2089 |                   0.219178 |
| bootstrap       | Q3                |     219 |            4.98667 |              4.98919 |    17.6294 |     21.964  |      15.362  |                   0.223744 |
| bootstrap       | Q4                |     219 |            6.59409 |              6.57356 |    19.8736 |     24.9447 |      16.8188 |                   0.324201 |
| bootstrap       | Q5 (Highest)      |     219 |            9.55099 |              8.99403 |    23.1074 |     29.0201 |      21.0101 |                   0.342466 |

## 5. Uncertainty Discrimination Performance
| ensemble_type   | error_threshold_definition                  |   error_threshold_ugm3 |   roc_auc |   pr_auc |   precision_at_top_10pct_spread |   recall_at_top_10pct_spread |   precision_at_top_20pct_spread |   recall_at_top_20pct_spread |   baseline_random_precision |
|:----------------|:--------------------------------------------|-----------------------:|----------:|---------:|--------------------------------:|-----------------------------:|--------------------------------:|-----------------------------:|----------------------------:|
| bootstrap       | Top 10% Absolute Error (>= 90th percentile) |                34.2803 |  0.756638 | 0.198691 |                        0.172727 |                     0.172727 |                        0.218182 |                     0.436364 |                    0.100365 |
| bootstrap       | Top 20% Absolute Error (>= 80th percentile) |                25.3478 |  0.719775 | 0.3326   |                        0.345455 |                     0.172727 |                        0.359091 |                     0.359091 |                    0.20073  |

## 6. Environmental Regime & Seasonal Uncertainty
### By Pollution Regime:
| pollution_regime   |   count |   mae_ugm3 |   rmse_ugm3 |   mean_ensemble_spread_ugm3 |   coverage_80pct |   coverage_90pct |   coverage_95pct |   mean_width_90pct_ugm3 |   winkler_score_90pct |
|:-------------------|--------:|-----------:|------------:|----------------------------:|-----------------:|-----------------:|-----------------:|------------------------:|----------------------:|
| Low                |     245 |    10.879  |     13.3588 |                     2.96877 |         0.167347 |         0.191837 |         0.220408 |                 8.94529 |               148.536 |
| Moderate           |     330 |    14.7724 |     18.3456 |                     4.66603 |         0.215152 |         0.266667 |         0.318182 |                14.0743  |               189.156 |
| High               |     330 |    20.0122 |     25.3762 |                     6.54019 |         0.215152 |         0.272727 |         0.306061 |                19.7438  |               258.635 |
| Extreme            |     191 |    23.6741 |     28.6221 |                     7.88725 |         0.204188 |         0.282723 |         0.335079 |                23.9801  |               292.677 |

### By Season:
| season       |   count |   mae_ugm3 |   rmse_ugm3 |   mean_spread_ugm3 |   median_spread_ugm3 |   coverage_90pct |   mean_width_90pct_ugm3 |   winkler_score_90pct |
|:-------------|--------:|-----------:|------------:|-------------------:|---------------------:|-----------------:|------------------------:|----------------------:|
| Winter       |     271 |    21.1396 |     26.744  |            7.24561 |              6.97332 |         0.343173 |                 21.9443 |               261.861 |
| Summer       |     276 |    14.5616 |     17.803  |            4.82864 |              4.40255 |         0.271739 |                 14.5718 |               184.925 |
| Monsoon      |     366 |    12.2828 |     14.8422 |            3.46886 |              2.76535 |         0.191257 |                 10.4487 |               165.375 |
| Post-Monsoon |     183 |    24.1678 |     29.7374 |            7.46449 |              7.21225 |         0.224044 |                 22.6164 |               314.387 |

### Year-to-Year Stability:
|   year |   count |   ensemble_mean_mae_ugm3 |   ensemble_mean_rmse_ugm3 |   ensemble_mean_r2 |   spearman_spread_error_corr |   coverage_90pct |   mean_width_90pct_ugm3 |   winkler_score_90pct |
|-------:|--------:|-------------------------:|--------------------------:|-------------------:|-----------------------------:|-----------------:|------------------------:|----------------------:|
|   2022 |     365 |                  17.9608 |                   23.3116 |           0.940417 |                     0.293441 |         0.260274 |                 17.9471 |               228.58  |
|   2023 |     365 |                  16.1932 |                   20.5791 |           0.959552 |                     0.237916 |         0.257534 |                 15.981  |               205.487 |
|   2024 |     366 |                  16.9394 |                   21.6047 |           0.948631 |                     0.306192 |         0.245902 |                 15.1585 |               223.031 |

### Extreme Pollution Stress Test:
| threshold_category                    |   threshold_value_ugm3 |   count |   mae_ugm3 |   rmse_ugm3 |   mean_spread_ugm3 |   coverage_80pct |   coverage_90pct |   coverage_95pct |   mean_width_90pct_ugm3 |   winkler_score_90pct |
|:--------------------------------------|-----------------------:|--------:|-----------:|------------:|-------------------:|-----------------:|-----------------:|-----------------:|------------------------:|----------------------:|
| Extreme Episodes (PM2.5 >= 150 µg/m³) |                    150 |     398 |    21.9873 |     27.4613 |            7.3621  |         0.226131 |         0.301508 |         0.346734 |                 22.3029 |               274.382 |
| Severe Episodes (PM2.5 >= 250 µg/m³)  |                    250 |     191 |    23.6741 |     28.6221 |            7.88725 |         0.204188 |         0.282723 |         0.335079 |                 23.9801 |               292.677 |

## 7. Ensemble Size Sensitivity
|   ensemble_size |   prediction_mae_ugm3 |   mean_spread_ugm3 |   coverage_90pct |   mpiw_90pct_ugm3 |   winkler_score_90pct |   spearman_spread_error_corr | status   |
|----------------:|----------------------:|-------------------:|-----------------:|------------------:|----------------------:|-----------------------------:|:---------|
|               5 |               17.0783 |            5.07247 |         0.166058 |           11.0944 |               254.44  |                     0.252012 | PASS     |
|              10 |               17.0833 |            5.21231 |         0.19708  |           13.8721 |               235.104 |                     0.278378 | PASS     |
|              20 |               17.0735 |            5.36152 |         0.237226 |           15.4637 |               224.522 |                     0.283016 | PASS     |
|              30 |               17.031  |            5.41229 |         0.254562 |           16.3611 |               219.036 |                     0.280451 | PASS     |

## 8. Representative Success & Failure Case Studies
| case_study_name                         | date       |   observed_pm25_ugm3 |   production_prediction_ugm3 |   ensemble_mean_ugm3 |   ensemble_spread_std_ugm3 |   interval_90pct_lower_ugm3 |   interval_90pct_upper_ugm3 |   absolute_error_ugm3 | season   | pollution_regime   | scientific_interpretation                                                                                            |
|:----------------------------------------|:-----------|---------------------:|-----------------------------:|---------------------:|---------------------------:|----------------------------:|----------------------------:|----------------------:|:---------|:-------------------|:---------------------------------------------------------------------------------------------------------------------|
| Low-Error / Low-Spread                  | 2022-07-11 |                44.07 |                      47.2133 |              46.9217 |                    2.12406 |                     43.0856 |                     49.7776 |               2.85173 | Monsoon  | Low                | Ideal Calibration: Low model dispersion accurately signals high prediction reliability.                              |
| High-Error / High-Spread                | 2022-01-11 |               357.67 |                     297.659  |             300.212  |                   10.6539  |                    286.718  |                    319.274  |              57.4575  | Winter   | Extreme            | Uncertainty Awareness: Wide ensemble spread successfully flags dynamic atmospheric volatility.                       |
| High-Error / Low-Spread (Overconfident) | 2022-12-06 |               275.73 |                     212.461  |             212.814  |                    4.80244 |                    205.697  |                    219.902  |              62.9156  | Winter   | Extreme            | Failure Mode (Underestimation): Ensemble agrees closely on an inaccurate forecast (structural epistemic blindspot).  |
| Low-Error / High-Spread (Conservative)  | 2022-01-12 |               211.5  |                     219.992  |             217.212  |                    8.47235 |                    208.805  |                    232.013  |               5.71242 | Winter   | High               | Failure Mode (Overestimation): Ensemble exhibits high variance despite accurate mean prediction.                     |
| Extreme Severe Episode (>= 250 µg/m³)   | 2022-01-01 |               262.12 |                     293.124  |             296.263  |                   12.3806  |                    276.465  |                    317.285  |              34.1429  | Winter   | Extreme            | Stress Test: Severe episodic conditions trigger wider spread but require residual calibration for complete coverage. |
| Winter Inversion Stagnation             | 2022-01-01 |               262.12 |                     293.124  |             296.263  |                   12.3806  |                    276.465  |                    317.285  |              34.1429  | Winter   | Extreme            | Seasonal Regime: High baseline dispersion driven by dynamic planetary boundary layer trapping.                       |

## 9. Leakage & Reproducibility Audit
| audit_check                                             | condition                                               |   violations_detected | status   | notes                                                                          |
|:--------------------------------------------------------|:--------------------------------------------------------|----------------------:|:---------|:-------------------------------------------------------------------------------|
| Chronological Walk-Forward Fold Integrity               | Folds strictly evaluate 2022, 2023, 2024 respectively   |                     0 | PASS     | No future evaluation observations leaked into prior training windows           |
| Monotonic Date Progression per Fold                     | Evaluation timestamps strictly increase chronologically |                     0 | PASS     | Zero temporal shuffling or lookahead permutations                              |
| Physical Lower-Bound Non-Negativity (Clipped Intervals) | lower_bound >= 0.0 µg/m³ for all clipped intervals      |                     0 | PASS     | All 9864 clipped intervals respect physical non-negative PM2.5 concentrations  |
| Interval Boundary Order (Lower <= Upper)                | lower_bound <= upper_bound for all intervals            |                     0 | PASS     | Zero boundary inversions detected across all methods and nominal levels        |
| Total Ensemble Leakage Violations                       | Total violations == 0                                   |                     0 | PASS     | Zero leakage violations confirmed across the entire Phase 6B ensemble pipeline |

## 10. Scientific Language Safeguards
> **MODEL / ENSEMBLE DISPERSION ≠ STATISTICAL PREDICTION UNCERTAINTY ≠ PHYSICAL ATMOSPHERIC UNCERTAINTY ≠ CAUSAL UNCERTAINTY**  
> Ensemble spread reflects model sensitivity and parameter variance under training perturbation; it is not a direct measure of physical atmospheric stochasticity or chemical transport uncertainty.

## 11. Final Decision & Phase 6C Readiness
- **Decision**: **`PARTIALLY INFORMATIVE`**
- **Findings**:
  1. Ensemble spread demonstrates a statistically significant positive rank correlation with actual prediction error (Spearman rho = `0.2805`), successfully discriminating high-error observations (ROC-AUC = `0.7566`).
  2. However, raw empirical ensemble quantiles remain under-dispersed during extreme pollution episodes (covering `30.15%` on extreme days under nominal 90%), confirming that raw ensemble spread requires residual scaling and formal calibration.
  3. **Phase 6C Recommendation**: Proceed to Phase 6C (Advanced Residual / Variance-Conditioned Localization) and Phase 6D (Conformal Prediction) to combine ensemble spread with calibrated prediction intervals.

---
**Status**: **`PHASE 6B COMPLETE`**
