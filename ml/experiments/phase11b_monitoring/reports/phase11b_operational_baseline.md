# AtmosIQ Phase 11B: Operational Baseline & Distribution Report

## Operational Monitoring Window
- **Data Policy**: CONTROLLED REPLAY / SIMULATED OPERATIONAL DATA (Locked Real Evaluation Partition: 2022-01-01 to 2024-12-31, N=1083 sequences)
- **Baseline Partition**: Real Historical Development Data (2020-01-01 to 2021-12-31)
- **Production Model**: `AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0`

---

## 1. Input Quality Baseline
- **Features Monitored**: 35 prediction-safe features
- **Missing Value Count**: 0
- **Infinite Value Count**: 0
- **Input Quality Status**: 35/35 Features Clean (PASS_CLEAN)

---

## 2. Feature Distribution Drift Baseline
Reusing certified Phase 10B PSI and Wasserstein distance methodology:
- **Green Drift Features** (PSI < 0.10): 22 / 35
- **Yellow Drift Features** (0.10 <= PSI <= 0.25): 11 / 35
- **Red Critical Drift Features** (PSI > 0.40): 0 / 35

---

## 3. Prediction Distribution Baseline
- **Baseline Mean**: 135.73 µg/m³ (std: 104.01)
- **Operational Replay Mean**: 138.35 µg/m³ (std: 106.39)
- **Prediction PSI**: 0.0689 (YELLOW_MODERATE_DRIFT)
- **Prediction Wasserstein Distance**: 5.0878
- **Extreme Forecasts (> 250 µg/m³)**: 23.4%

---

## 4. Calibration & Uncertainty Baseline
- **Evaluation Samples**: 1083
- **Replay Stream MAE**: 36.399 µg/m³
- **Replay Stream RMSE**: 48.348 µg/m³
- **Residual Bias**: -4.639 µg/m³ (Calibration offset: -5.06 µg/m³)
- **Conformal 90% Interval Width**: ±95.66 µg/m³
- **Observed Empirical Coverage**: **93.17%** (Target: 90.0%)
- **Coverage Status**: PASS_WITHIN_TARGET
