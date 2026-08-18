# AtmosIQ Phase 9C: Model Hardening, Calibration & Uncertainty Report

## 1. Candidate Model Hardening Summary
Phase 9C finalized parameter manifests, fitted calibration bias corrections strictly on the 2020–2021 development validation fold, and established conformal prediction intervals.

### Candidate Comparison Table
| candidate_id         | model_version                                        | architecture   |   augmentation_ratio | governance_role      | production_eligibility   |   parameter_count |   uncalibrated_test_mae |   calibrated_test_mae |   calibrated_test_rmse |   calibrated_test_r2 |   pearson_r |   extreme_event_mae |   interval_coverage_90 |   average_interval_width |   calibration_bias_offset |
|:---------------------|:-----------------------------------------------------|:---------------|---------------------:|:---------------------|:-------------------------|------------------:|------------------------:|----------------------:|-----------------------:|---------------------:|------------:|--------------------:|-----------------------:|-------------------------:|--------------------------:|
| TCN_50pct_RESEARCH   | AtmosIQ_DL_TCN_CAL07_50_RESEARCH_v1.0.0              | TCN            |                 0.5  | RESEARCH_CANDIDATE   | RESTRICTED               |               865 |                 36.5778 |               35.9121 |                47.5543 |             0.758769 |    0.891853 |             43.9136 |               0.935305 |                  168.947 |                  -4.45506 |
| TCN_25pct_PRODUCTION | AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0  | TCN            |                 0.25 | PRODUCTION_CANDIDATE | PRODUCTION_ELIGIBLE      |               865 |                 37.811  |               37.099  |                49.0658 |             0.743189 |    0.888542 |             43.787  |               0.933457 |                  170.463 |                  -5.0592  |
| LSTM_25pct_FALLBACK  | AtmosIQ_DL_LSTM_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0 | LSTM           |                 0.25 | PRODUCTION_CANDIDATE | PRODUCTION_ELIGIBLE      |               849 |                133.682  |               92.915  |               103.956  |            -0.152802 |   -0.300914 |            143.254  |               1        |                  497.261 |                -167.249   |

---

## 2. Validation Bias Calibration Results (`phase9cd_calibration_results.csv`)
| candidate_id         |   calibration_bias_fitted |   raw_test_mae |   calibrated_test_mae |   mae_improvement |   calibrated_test_rmse |
|:---------------------|--------------------------:|---------------:|----------------------:|------------------:|-----------------------:|
| TCN_50pct_RESEARCH   |                  -4.45506 |        36.5778 |               35.9121 |          0.665691 |                47.5543 |
| TCN_25pct_PRODUCTION |                  -5.0592  |        37.811  |               37.099  |          0.712025 |                49.0658 |
| LSTM_25pct_FALLBACK  |                -167.249   |       133.682  |               92.915  |         40.7665   |               103.956  |

---

## 3. Conformal Prediction Interval Uncertainty Results (`phase9cd_uncertainty_results.csv`)
| candidate_id         |   conformal_bound_80 |   conformal_bound_90 |   conformal_bound_95 |   interval_coverage |   average_interval_width |   extreme_interval_coverage | uncertainty_type                         |
|:---------------------|---------------------:|---------------------:|---------------------:|--------------------:|-------------------------:|----------------------------:|:-----------------------------------------|
| TCN_50pct_RESEARCH   |              60.9431 |              94.2748 |              113.908 |            0.935305 |                  168.947 |                    0.911111 | Empirical Conformal (Aleatoric Residual) |
| TCN_25pct_PRODUCTION |              63.9219 |              95.6647 |              117.5   |            0.933457 |                  170.463 |                    0.922222 | Empirical Conformal (Aleatoric Residual) |
| LSTM_25pct_FALLBACK  |             289.855  |             320.771  |              354.016 |            1        |                  497.261 |                    1        | Empirical Conformal (Aleatoric Residual) |
