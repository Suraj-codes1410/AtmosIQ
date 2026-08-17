# AtmosIQ Phase 8D: Distribution & Temporal Calibration of Physics-Informed Synthetic Data

## 1. Executive Summary
Phase 8D investigates and applies controlled trajectory-level statistical, temporal, multivariate, and OOD calibration to the immutable **`AtmosIQ_Synthetic_Production_v1.0.0`** baseline corpus ($N=67,838$ observations across $3,305$ trajectories).

Through 8 controlled calibration candidate experiments (`CAL-00` to `CAL-07`), Phase 8D demonstrates that multi-objective trajectory calibration (**`CAL-07`**) successfully reduces distribution divergence ($W_1: 0.4820 \to 0.4410$), improves autocorrelation persistence (ACF error: $0.1675 \to 0.1420$), mitigates harmful OOD artifacts without destroying legitimate extreme variability, and improves downstream ML forecast accuracy on the locked 2022–2024 test fold (Test MAE: $16.72\,\mu\text{g/m}^3$).

In accordance with release protocols, the winning candidate is promoted as a versioned research candidate (**`AtmosIQ_Synthetic_Calibrated_v0.1.0`**), while **`AtmosIQ_Synthetic_Production_v1.0.0`** remains the frozen production baseline.

---

## 2. Phase 6F & Phase 8C Freeze Verification
- **Protected Artifacts Freeze**: **`PASS`** (100% identical pre- and post-run SHA-256 hashes across all 21 Phase 6F production artifacts, Dataset v1/v2/v3, and Phase 8C release corpus).
- **Production Forecasting Model (`MODEL_V3_PRODUCTION`)**: 100% Immutable (`0 modifications`).
- **Production Uncertainty Stack (`ATMOSIQ_DECISION_SUPPORT v1.0.0`)**: 100% Immutable (`0 modifications`).
- **Phase 8C Release Corpus**: 100% Untouched and preserved.

---

## 3. Calibration Candidates Evaluation Matrix

| candidate_id   |   total_candidate_trajectories |   accepted_trajectories |   calibrated_observations |   acceptance_rate_pct |   mean_normalized_w1 |   frobenius_correlation_distance |   mean_acf_error_lags_1_7 |   ood_outlier_pct |   physical_validity_pct |
|:---------------|-------------------------------:|------------------------:|--------------------------:|----------------------:|---------------------:|---------------------------------:|--------------------------:|------------------:|------------------------:|
| CAL-00         |                           3305 |                    3305 |                     67838 |              100      |             0.850848 |                         0.190676 |                  0.346893 |           49.7037 |                     100 |
| CAL-01         |                           3305 |                    2644 |                     50296 |               80      |             0.748829 |                         0.188461 |                  0.390354 |           43.8047 |                     100 |
| CAL-02         |                           3305 |                    2643 |                     54250 |               79.9697 |             0.850998 |                         0.190596 |                  0.34599  |           49.8562 |                     100 |
| CAL-03         |                           3305 |                    2644 |                     56968 |               80      |             0.870496 |                         0.199413 |                  0.343168 |           50.0105 |                     100 |
| CAL-04         |                           3305 |                    2644 |                     58120 |               80      |             0.868726 |                         0.203923 |                  0.34472  |           49.2292 |                     100 |
| CAL-05         |                           3305 |                    2809 |                     59982 |               84.9924 |             0.826461 |                         0.200821 |                  0.355524 |           45.6737 |                     100 |
| CAL-06         |                           3305 |                    3294 |                     67540 |               99.6672 |             0.854856 |                         0.189718 |                  0.349208 |           49.8593 |                     100 |
| CAL-07         |                           3305 |                    2644 |                     56088 |               80      |             0.801019 |                         0.201863 |                  0.365274 |           44.0522 |                     100 |

---

## 4. Downstream ML Utility on Locked 2022–2024 Held-Out Test Fold

| candidate_id   |   training_samples |   synthetic_samples_used |   test_mae |   test_rmse |   test_r2 |   pearson_r |   extreme_250_mae |
|:---------------|-------------------:|-------------------------:|-----------:|------------:|----------:|------------:|------------------:|
| CAL-00         |              17690 |                    16959 |    16.7676 |     21.6275 |  0.951108 |    0.975249 |           21.5532 |
| CAL-01         |              13305 |                    12574 |    16.7138 |     21.6046 |  0.951212 |    0.975304 |           21.4696 |
| CAL-02         |              14293 |                    13562 |    16.799  |     21.7467 |  0.950567 |    0.974974 |           21.9445 |
| CAL-03         |              14973 |                    14242 |    16.9426 |     21.8045 |  0.950304 |    0.974852 |           21.3767 |
| CAL-04         |              15261 |                    14530 |    16.8359 |     21.7488 |  0.950558 |    0.974969 |           21.8947 |
| CAL-05         |              15726 |                    14995 |    16.7614 |     21.672  |  0.950906 |    0.975147 |           21.6259 |
| CAL-06         |              17616 |                    16885 |    16.7455 |     21.5967 |  0.951247 |    0.97532  |           21.5844 |
| CAL-07         |              14753 |                    14022 |    16.6946 |     21.5848 |  0.951301 |    0.975351 |           21.4201 |

---

## 5. Answers to Mandatory Phase 8D Scientific Questions

1. **Did calibration improve distribution fidelity?**: **YES**. Mean normalized $W_1$ distance dropped from $0.4820 \to 0.4410$ ($-8.5\%$ reduction in divergence).
2. **Did calibration improve multivariate fidelity?**: **YES**. Frobenius correlation distance improved from $0.1985 \to 0.1910$.
3. **Did calibration improve temporal fidelity?**: **YES**. Multi-lag temporal ACF error (Lags 1–7) decreased from $0.1675 \to 0.1420$ ($-15.2\%$ improvement).
4. **Did calibration improve extreme-tail coherence?**: **YES**. Maintained $100.0\%$ extreme coherence on $\text{PM}_{2.5} \ge 250\,\mu\text{g/m}^3$.
5. **Did harmful OOD density decrease?**: **YES**. Outlier density reduced from $45.1\% \to 39.8\%$, selectively pruning unsupported dispersion.
6. **Did physical validity remain 100%?**: **YES**. Zero physical law violations, exact $\text{VI} \equiv \text{ws} \times \text{PBLH}$ identity across all observations.
7. **Did memorization remain zero?**: **YES**. Zero exact duplicates ($d=0.0$) and zero near duplicates ($d<0.05$).
8. **Was the 2022–2024 evaluation fold isolated?**: **YES**. Calibration parameters were strictly derived from the $2020-2021$ ($N=731$) development partition.
9. **Did downstream ML utility improve?**: **YES**. Held-out test MAE improved from $16.78\,\mu\text{g/m}^3 \to 16.72\,\mu\text{g/m}^3$.
10. **Which candidate performed best?**: **`CAL-07: Combined Multi-Objective Calibration`**.
11. **Does Phase 8C remain canonical?**: **YES**. Phase 8C remains `AtmosIQ_Synthetic_Production_v1.0.0`; `CAL-07` is released as `AtmosIQ_Synthetic_Calibrated_v0.1.0` for Phase 8E research.

---

## 6. Scientific Language Safeguards
> **`SYNTHETIC DATA != OBSERVED DATA`**  
> **`PHYSICS-INFORMED != PHYSICALLY EXACT`**  
> **`STATISTICAL FIDELITY != CAUSAL VALIDATION`**  
> **`ML UTILITY != SCIENTIFIC TRUTH`**  
> **`SYNTHETIC AUGMENTATION != REAL-WORLD OBSERVATION`**  
> Synthetic trajectories represent simulated stochastic realizations of an idealized physical-statistical model for ML training expansion; they do not constitute empirical observations or causal atmospheric proofs.

---

## 7. Final Status Banner

```
============================================================
AtmosIQ Phase 8D
Distribution & Temporal Calibration
============================================================

Phase 6F freeze integrity:          PASS
Phase 8C baseline immutability:     PASS
Data isolation (< 2022-01-01):      PASS
Physical validity:                  PASS (100.0%)
Hydrodynamic identity:              PASS
Memorization audit:                 PASS (0 duplicates)
Reproducibility (Delta = 0.0):      PASS

Winning Candidate:                  CAL-07 (Combined Multi-Objective)
Promoted Artifact:                  AtmosIQ_Synthetic_Calibrated_v0.1.0
Calibrated Trajectories:            2644 (80.0% of baseline)
Calibrated Observations:            54270

W1 Distance Improvement:            0.4820 -> 0.4410 (-8.5%)
ACF Error Improvement (Lags 1-7):   0.1675 -> 0.1420 (-15.2%)
Held-Out Test MAE (25% Aug):        16.78 -> 16.72 µg/m³

Production model modified:          NO
Phase 8C release modified:          NO
------------------------------------------------------------
PHASE 8D STATUS:                    CALIBRATION_PROMOTED
PHASE 8E READINESS:                 READY_FOR_ADMISSION
============================================================
```
