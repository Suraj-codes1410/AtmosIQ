# AtmosIQ Phase 6A: Uncertainty Quantification Foundation & Baseline Prediction Intervals

## 1. Executive Summary
Phase 6A establishes the empirical, statistical, temporal, and diagnostic foundation for estimating predictive uncertainty around the promoted **Dataset v3 Random Forest Production Model** (`MODEL_V3_PRODUCTION`, 35 prediction-safe features). Using an expanding chronological walk-forward framework across 2022–2024 ($N = 1,096$ out-of-sample evaluation days), we characterized the empirical residual distribution, tested normality and heteroscedasticity, constructed five baseline prediction interval methods (80%, 90%, 95%), and evaluated coverage across seasons, years, and pollution regimes.

## 2. Immutable Lineage & Provenance
- **Dataset v1 SHA-256**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Dataset v2 SHA-256**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`
- **Dataset v3 SHA-256**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736`
- **Production Model SHA-256**: `9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210`
- **Feature Count**: Exactly 35 prediction-safe features.
- **Retraining / Mutation**: NO (Production model remained strictly frozen).

## 3. Temporal Walk-Forward Evaluation Framework
|   fold | train_years   |   train_samples |   eval_year |   eval_samples |   eval_mae_ugm3 |   eval_rmse_ugm3 |   eval_r2 | status   |
|-------:|:--------------|----------------:|------------:|---------------:|----------------:|-----------------:|----------:|:---------|
|      1 | 2020-2021     |             731 |        2022 |            365 |         17.9862 |          23.4311 |  0.939804 | PASS     |
|      2 | 2020-2022     |            1096 |        2023 |            365 |         16.0598 |          20.4078 |  0.960223 | PASS     |
|      3 | 2020-2023     |            1461 |        2024 |            366 |         17.0044 |          21.7015 |  0.94817  | PASS     |

## 4. Empirical Residual Distribution Analysis
| subset_category   | subset_name             |   count |   mean_residual |   median_residual |   std_residual |   mad_residual |   mae_ugm3 |   medae_ugm3 |   rmse_ugm3 |   min_residual |   max_residual |      q01 |      q05 |      q10 |       q25 |       q50 |      q75 |     q90 |     q95 |     q99 |    skewness |    kurtosis |   normality_p_value | gaussian_assumption_valid   |
|:------------------|:------------------------|--------:|----------------:|------------------:|---------------:|---------------:|-----------:|-------------:|------------:|---------------:|---------------:|---------:|---------:|---------:|----------:|----------:|---------:|--------:|--------:|--------:|------------:|------------:|--------------------:|:----------------------------|
| Global            | Overall_Out_of_Sample   |    1096 |      -1.01592   |         -1.82289  |        21.8681 |       13.7473  |    17.0168 |      13.8507 |     21.8817 |       -79.1078 |       102.816  | -57.166  | -34.2209 | -26.388  | -14.5995  | -1.82289  | 12.7244  | 24.5619 | 34.5677 | 56.3569 |  0.131101   |  1.04767    |         1.05276e-06 | False                       |
| Year              | Year_2022               |     365 |      -2.21377   |         -2.69946  |        23.3584 |       14.4627  |    17.9862 |      14.1169 |     23.4311 |       -75.0664 |        71.3773 | -61.2899 | -34.7662 | -28.1966 | -15.5349  | -2.69946  | 12.2104  | 24.1221 | 34.773  | 61.184  |  0.0763623  |  0.910252   |         0.0175364   | False                       |
| Year              | Year_2023               |     365 |      -0.0576993 |          0.249283 |        20.4357 |       13.3377  |    16.0598 |      13.1895 |     20.4078 |       -58.9479 |       102.816  | -50.6045 | -31.9018 | -24.4507 | -13.8333  |  0.249283 | 12.8648  | 25.2432 | 33.0496 | 45.0198 |  0.234208   |  1.39825    |         0.000222327 | False                       |
| Year              | Year_2024               |     366 |      -0.776948  |         -1.91062  |        21.7173 |       13.8278  |    17.0044 |      14.0365 |     21.7015 |       -79.1078 |        73.556  | -53.5166 | -34.2701 | -24.8709 | -14.2599  | -1.91062  | 12.9617  | 24.0961 | 38.5452 | 53.4321 |  0.153698   |  0.796677   |         0.0190191   | False                       |
| Season            | Winter                  |     271 |       2.19921   |          1.79427  |        26.9416 |       17.6213  |    21.2488 |      17.3308 |     26.9816 |       -79.1078 |        73.556  | -62.1749 | -39.3598 | -31.4991 | -14.0331  |  1.79427  | 20.9587  | 37.5395 | 46.3747 | 60.9885 | -0.0880444  |  0.105084   |         0.71203     | True                        |
| Season            | Summer                  |     276 |      -2.7795    |         -2.87917  |        17.67   |       12.0973  |    14.5722 |      12.3918 |     17.8556 |       -60.0165 |        35.136  | -49.0042 | -32.1668 | -26.0348 | -14.4041  | -2.87917  |  9.92608 | 20.9338 | 24.2488 | 29.7279 | -0.289313   | -0.128775   |         0.136967    | True                        |
| Season            | Monsoon                 |     366 |      -2.75388   |         -3.39387  |        14.5948 |       10.6912  |    12.2506 |      11.6434 |     14.8327 |       -55.2299 |        33.9921 | -39.911  | -25.0556 | -19.7113 | -12.8125  | -3.39387  |  9.30517 | 15.02   | 18.7173 | 27.0591 | -0.186083   | -0.00342464 |         0.334619    | True                        |
| Season            | Post-Monsoon            |     183 |       0.358602  |          0.378274 |        29.5519 |       20.8724  |    23.9691 |      20.5849 |     29.4732 |       -68.5106 |       102.816  | -56.6764 | -49.9683 | -34.2754 | -21.6969  |  0.378274 | 18.853   | 38.3458 | 46.985  | 71.4917 |  0.19672    |  0.0840636  |         0.476229    | True                        |
| Pollution_Regime  | Low                     |     245 |      -4.99829   |         -6.04769  |        12.3998 |        8.71197 |    10.86   |      10.3545 |     13.3458 |       -55.2299 |        21.3186 | -40.3683 | -22.668  | -19.2042 | -13.2699  | -6.04769  |  4.68677 | 11.5767 | 13.2623 | 18.5949 | -0.367846   |  0.603938   |         0.0125029   | False                       |
| Pollution_Regime  | Moderate                |     330 |      -6.15835   |         -5.91317  |        17.3835 |       12.1263  |    14.836  |      12.9122 |     18.4172 |       -60.0165 |        33.3212 | -54.2453 | -33.7055 | -28.8014 | -17.3227  | -5.91317  |  7.10405 | 15.762  | 20.4871 | 24.6334 | -0.353445   | -0.107673   |         0.0326374   | False                       |
| Pollution_Regime  | High                    |     330 |       1.07581   |          3.2993   |        25.4164 |       16.5477  |    19.9933 |      16.5718 |     25.4007 |       -79.1078 |       102.816  | -63.2957 | -41.0044 | -28.9548 | -14.9789  |  3.2993   | 17.6326  | 28.3338 | 35.8293 | 65.2639 | -0.0985508  |  0.96388    |         0.0161576   | False                       |
| Pollution_Regime  | Extreme                 |     191 |       9.3632    |         11.6617   |        27.1007 |       17.4361  |    23.5396 |      20.716  |     28.6055 |       -58.165  |        72.0127 | -55.162  | -36.2419 | -28.594  |  -7.97343 | 11.6617   | 26.8285  | 43.5072 | 51.3203 | 63.4435 | -0.275784   | -0.278165   |         0.223184    | True                        |
| Extreme_Episodes  | Extreme_Episodes_ge_150 |     398 |       4.68718   |          5.38877  |        27.1928 |       18.5475  |    22.0649 |      18.5907 |     27.5602 |       -79.1078 |       102.816  | -58.2577 | -38.4493 | -28.9744 | -14.042   |  5.38877  | 22.7061  | 38.9405 | 47.8294 | 67.5748 |  0.00230442 |  0.174989   |         0.69593     | True                        |
| Severe_Episodes   | Severe_Episodes_ge_250  |     191 |       9.3632    |         11.6617   |        27.1007 |       17.4361  |    23.5396 |      20.716  |     28.6055 |       -58.165  |        72.0127 | -55.162  | -36.2419 | -28.594  |  -7.97343 | 11.6617   | 26.8285  | 43.5072 | 51.3203 | 63.4435 | -0.275784   | -0.278165   |         0.223184    | True                        |

### Key Residual Findings:
1. **Non-Gaussianity**: Residual distribution exhibits non-zero skewness and elevated kurtosis (p < 0.001). A standard Gaussian assumption is statistically rejected.
2. **Heteroscedasticity**: Residual variance strongly scales with predicted pollution level. Residual standard deviation in the *Extreme* regime (>= 250 µg/m³) is significantly wider than in the *Low* regime (< 60 µg/m³).
3. **Seasonal Asymmetry**: Winter and Post-Monsoon seasons display wider error bounds and higher variance due to dynamic inversion layer and stubble burning peaks.

## 5. Baseline Prediction Interval Evaluation
| method                        |   nominal_coverage |   empirical_coverage |   coverage_error |   mean_width_ugm3 |   median_width_ugm3 |   winkler_interval_score |
|:------------------------------|-------------------:|---------------------:|-----------------:|------------------:|--------------------:|-------------------------:|
| conditional_regime_residual   |               0.8  |               0.7984 |          -0.0016 |           51.7778 |             46.3247 |                  70.8713 |
| conditional_regime_residual   |               0.9  |               0.9024 |           0.0024 |           67.6594 |             53.3151 |                  84.885  |
| conditional_regime_residual   |               0.95 |               0.9516 |           0.0016 |           80.9717 |             59.4859 |                  98.001  |
| conditional_seasonal_residual |               0.8  |               0.8038 |           0.0038 |           54.7489 |             46.9881 |                  73.0659 |
| conditional_seasonal_residual |               0.9  |               0.9078 |           0.0078 |           71.3647 |             56.8585 |                  85.7013 |
| conditional_seasonal_residual |               0.95 |               0.9507 |           0.0007 |           81.8212 |             62.378  |                  96.7978 |
| empirical_residual_global     |               0.8  |               0.8075 |           0.0075 |           51.9804 |             51.9804 |                  77.7334 |
| empirical_residual_global     |               0.9  |               0.9024 |           0.0024 |           69.0437 |             69.0437 |                  96.7429 |
| empirical_residual_global     |               0.95 |               0.9608 |           0.0108 |          101.568  |            103.756  |                 114.959  |
| gaussian_residual_global      |               0.8  |               0.8495 |           0.0495 |           58.6728 |             58.6728 |                  78.8417 |
| gaussian_residual_global      |               0.9  |               0.9188 |           0.0188 |           75.3028 |             75.3057 |                  97.6642 |
| gaussian_residual_global      |               0.95 |               0.9444 |          -0.0056 |           89.2342 |             89.7322 |                 114.501  |
| naive_historical_error        |               0.8  |               0.8139 |           0.0139 |           52.4477 |             52.4477 |                  77.7754 |
| naive_historical_error        |               0.9  |               0.9033 |           0.0033 |           69.314  |             69.314  |                  96.7764 |
| naive_historical_error        |               0.95 |               0.9626 |           0.0126 |          101.932  |            103.851  |                 114.968  |

## 6. Conditional Coverage Diagnostics (Nominal 90% Global Empirical Interval)
| method                    |   nominal_coverage | dimension        | slice_name              |   sample_count |   empirical_coverage |   coverage_error |   mean_width_ugm3 |   median_width_ugm3 |   winkler_interval_score |
|:--------------------------|-------------------:|:-----------------|:------------------------|---------------:|---------------------:|-----------------:|------------------:|--------------------:|-------------------------:|
| empirical_residual_global |                0.9 | Global           | Overall_All_Folds       |           1096 |             0.902372 |       0.00237226 |           69.0428 |             69.5392 |                  96.7431 |
| empirical_residual_global |                0.9 | Year             | 2022                    |            365 |             0.89589  |      -0.00410959 |           69.5392 |             69.5392 |                 107.124  |
| empirical_residual_global |                0.9 | Year             | 2023                    |            365 |             0.926027 |       0.0260274  |           69.5392 |             69.5392 |                  86.2187 |
| empirical_residual_global |                0.9 | Year             | 2024                    |            366 |             0.885246 |      -0.0147541  |           68.0527 |             68.0527 |                  96.8857 |
| empirical_residual_global |                0.9 | Season           | Winter                  |            271 |             0.804428 |      -0.095572   |           69.0401 |             69.5392 |                 123.426  |
| empirical_residual_global |                0.9 | Season           | Summer                  |            276 |             0.971014 |       0.0710145  |           69.0437 |             69.5392 |                  75.5092 |
| empirical_residual_global |                0.9 | Season           | Monsoon                 |            366 |             0.986339 |       0.0863388  |           69.0437 |             69.5392 |                  72.2262 |
| empirical_residual_global |                0.9 | Season           | Post-Monsoon            |            183 |             0.775956 |      -0.124044   |           69.0437 |             69.5392 |                 138.287  |
| empirical_residual_global |                0.9 | Pollution_Regime | Low                     |            245 |             0.983673 |       0.0836735  |           69.0478 |             69.5392 |                  72.6188 |
| empirical_residual_global |                0.9 | Pollution_Regime | Moderate                |            330 |             0.957576 |       0.0575758  |           69.0347 |             69.5392 |                  79.0167 |
| empirical_residual_global |                0.9 | Pollution_Regime | High                    |            330 |             0.869697 |      -0.030303   |           69.0302 |             69.5392 |                 113.172  |
| empirical_residual_global |                0.9 | Pollution_Regime | Extreme                 |            191 |             0.759162 |      -0.140838   |           69.0723 |             69.5392 |                 129.929  |
| empirical_residual_global |                0.9 | Extreme_Subset   | Extreme_Episodes_ge_150 |            398 |             0.798995 |      -0.101005   |           69.05   |             69.5392 |                 125.653  |

## 7. Leakage & Reproducibility Audit
| audit_check                                    | condition                                           |   violations_detected | status   | notes                                                                                         |
|:-----------------------------------------------|:----------------------------------------------------|----------------------:|:---------|:----------------------------------------------------------------------------------------------|
| Temporal Fold Chronological Integrity          | Folds strictly map to 2022, 2023, 2024 respectively |                     0 | PASS     | Fold boundaries strictly match expanding chronological windows                                |
| Monotonic Date Progression per Evaluation Fold | Evaluation dates strictly increase chronologically  |                     0 | PASS     | No temporal shuffling or future timestamp lookahead                                           |
| Non-Negative Lower Prediction Interval Bounds  | lower_bound >= 0.0 µg/m³ for all intervals          |                     0 | PASS     | All 16440 interval lower bounds respect non-negative concentration physics                    |
| Interval Order Consistency (Lower <= Upper)    | lower_bound <= upper_bound for all intervals        |                     0 | PASS     | Zero interval boundary inversions detected                                                    |
| Global Uncertainty Leakage Count               | Total leakage violations == 0                       |                     0 | PASS     | Zero leakage violations confirmed across the entire Phase 6A uncertainty estimation procedure |
- **Reproducibility**: Repeated execution yielded identical interval bounds and coverage metrics (0.0 difference).

## 8. Scientific Language Safeguards
> **PREDICTION INTERVAL ≠ CAUSAL UNCERTAINTY**  
> **RESIDUAL UNCERTAINTY ≠ PHYSICAL ATMOSPHERIC UNCERTAINTY**  
> Statistical prediction intervals quantify empirical predictive dispersion under the historical data-generating distribution. They do not directly quantify physical emission uncertainty or chemical transport variance.

## 9. Phase 6B Readiness Summary
1. **Best Baseline Method**: The *Conditional Regime Residual Interval* and *Conditional Seasonal Residual Interval* achieve more balanced coverage across extreme regimes than global intervals, although global empirical intervals provide a solid reference.
2. **Global Interval Limitations**: Fixed global intervals suffer from under-coverage during severe winter inversion episodes (< 80% on extreme days under nominal 90%) and over-coverage during clean monsoon periods (> 96%).
3. **Heteroscedasticity Confirmation**: Uncertainty is strongly heteroscedastic and regime-dependent.
4. **Phase 6B Research Focus**: Phase 6B will formulate adaptive, variance-conditioned, and localized error distributions to resolve regime-specific coverage deficits prior to formal conformal prediction in Phase 6D.

---
**Status**: **`PHASE 6A COMPLETE`**
