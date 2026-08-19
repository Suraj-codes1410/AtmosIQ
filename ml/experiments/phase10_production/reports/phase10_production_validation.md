# AtmosIQ Phase 10: Production Validation & Walk-Forward Backtesting Report

## 1. Walk-Forward Rolling-Origin Temporal Validation (`phase10_walkforward_results.csv`)
| fold_id     | train_period             | val_period               |   sequences_count |     mae |    rmse |       r2 |   pearson_r |   extreme_mae |   extreme_rmse |   extreme_count |   prediction_bias |   coverage_90 |   interval_width_90 |
|:------------|:-------------------------|:-------------------------|------------------:|--------:|--------:|---------:|------------:|--------------:|---------------:|----------------:|------------------:|--------------:|--------------------:|
| WF_Window_A | 2020-01-01 to 2020-12-31 | 2021-01-01 to 2021-06-30 |               167 | 34.8808 | 44.5837 | 0.58535  |    0.822859 |       51.8503 |        58.5354 |              14 |         -13.6987  |      0.982036 |             180.434 |
| WF_Window_B | 2020-01-01 to 2021-06-30 | 2021-07-01 to 2021-12-31 |               170 | 38.7076 | 55.2561 | 0.767514 |    0.919329 |       47.4495 |        57.0078 |              43 |           8.23547 |      0.9      |             161.492 |
| WF_Window_C | 2020-01-01 to 2021-12-31 | 2022-01-01 to 2022-12-31 |               351 | 38.9071 | 50.9455 | 0.693553 |    0.868465 |       48.2973 |        56.7113 |              48 |          -3.30837 |      0.923077 |             169.606 |
| WF_Window_D | 2020-01-01 to 2022-12-31 | 2023-01-01 to 2024-12-31 |               717 | 35.529  | 47.2565 | 0.762795 |    0.895588 |       42.4614 |        53.778  |             122 |          -6.1485  |      0.941423 |             170.85  |

---

## 2. Temporal Leakage & Preprocessing Isolation Audit (`phase10_walkforward_leakage_audit.csv`)
| fold_id     | train_start   | train_end   | val_start   | val_end    | max_train_date   | min_val_date   | temporal_firewall_passed   |   train_observations |   val_observations | scaler_fitted_on         | scaler_leakage   | target_leakage   | status   |
|:------------|:--------------|:------------|:------------|:-----------|:-----------------|:---------------|:---------------------------|---------------------:|-------------------:|:-------------------------|:-----------------|:-----------------|:---------|
| WF_Window_A | 2020-01-01    | 2020-12-31  | 2021-01-01  | 2021-06-30 | 2020-12-31       | 2021-01-01     | True                       |                  366 |                181 | 2020-01-01 to 2020-12-31 | NONE             | NONE             | PASS     |
| WF_Window_B | 2020-01-01    | 2021-06-30  | 2021-07-01  | 2021-12-31 | 2021-06-30       | 2021-07-01     | True                       |                  547 |                184 | 2020-01-01 to 2021-06-30 | NONE             | NONE             | PASS     |
| WF_Window_C | 2020-01-01    | 2021-12-31  | 2022-01-01  | 2022-12-31 | 2021-12-31       | 2022-01-01     | True                       |                  731 |                365 | 2020-01-01 to 2021-12-31 | NONE             | NONE             | PASS     |
| WF_Window_D | 2020-01-01    | 2022-12-31  | 2023-01-01  | 2024-12-31 | 2022-12-31       | 2023-01-01     | True                       |                 1096 |                731 | 2020-01-01 to 2022-12-31 | NONE             | NONE             | PASS     |

---

## 3. Seasonal Breakdown (`phase10_temporal_breakdown.csv`)
| season       |   observations |     mae |    rmse |      bias |
|:-------------|---------------:|--------:|--------:|----------:|
| Winter       |            319 | 53.1287 | 63.42   |   3.71493 |
| Summer       |            368 | 27.7249 | 35.2236 | -12.5197  |
| Monsoon      |            474 | 22.9202 | 30.8878 |  -9.03527 |
| Post-Monsoon |            244 | 55.4143 | 69.2859 |   5.11285 |

---

## 4. Pollution Regime Breakdown (`phase10_regime_breakdown.csv`)
| regime       |   observations |     mae |    rmse |     bias |
|:-------------|---------------:|--------:|--------:|---------:|
| Good         |             56 | 11.9101 | 16.3607 | 11.0336  |
| Satisfactory |            256 | 15.5089 | 18.7353 | -5.26795 |
| Moderate     |            440 | 28.2116 | 35.718  | -6.52549 |
| Poor/Severe  |            426 | 56.8567 | 68.5543 | -3.41525 |
| Emergency    |            227 | 45.2194 | 55.3294 | -6.16981 |

---

## 5. Operational Input Robustness Audit (`phase10_input_robustness.csv`)
| input_case                       | expected_behavior                                    | actual_behavior      | pass_fail   |
|:---------------------------------|:-----------------------------------------------------|:---------------------|:------------|
| Missing Feature (D=34)           | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| Extra Feature (D=36)             | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| NaN Value in Tensor              | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| Inf Value in Tensor              | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| Wrong Sequence Length (W=7)      | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| Wrong Sequence Length (W=28)     | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| 2D Tensor Dimension (B, D)       | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| 4D Tensor Dimension (B, 1, W, D) | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| Non-Numpy Object Input           | Reject Safely with Validation Error or Finite Output | PASS_SAFELY_REJECTED | PASS        |
| Empty Input Tensor (B=0)         | Reject Safely with Validation Error or Finite Output | PASS_SAFE_EXECUTION  | PASS        |
| Extreme Out-of-Range (>1e6)      | Reject Safely with Validation Error or Finite Output | PASS_SAFE_EXECUTION  | PASS        |
| Reordered Feature Schema         | Reject Safely with Validation Error or Finite Output | PASS_SAFE_EXECUTION  | PASS        |
