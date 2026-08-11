# AtmosIQ Phase 3A: Target & Temporal Leakage Audit

## 1. Audit Overview & Verification Protocol

Target and temporal leakage are common failure modes in time-series machine learning. AtmosIQ performs a 5-step automated and manual audit before any model fitting is authorized.

- **Audit Date**: 2026-08-11
- **Dataset Evaluated**: `ml/data/modeling/v1/feature_dataset_frozen.csv`
- **Predictor Set**: 254 predictor features
- **Audit Outcome**: **PASS**

---

## 2. Audit Verification Matrix

| Audit Check | Risk Description | Prevention Mechanism | Status |
|---|---|---|---|
| **1. Current-day Target Leakage** | `pm25(t)` appearing as a predictor | `pm25` is isolated as target $y$. Lags use `shift(1)` or greater. | **PASS** |
| **2. Unshifted Rolling Window Leakage** | `pm25(t)` included in rolling mean | All target rolling stats shift input by 1 day before rolling calculation ($[t-W, t-1]$). | **PASS** |
| **3. Same-Day Measurement Exclusion** | Unforecasted same-day measurements | `SAME_DAY_FEATURE` columns are classified and marked `prediction_safe: false` for 24h prediction cutoff. | **PASS** |
| **4. Global Preprocessing Leakage** | Scaler fitted across full 731 rows | Global fitting prohibited. Scalers must be fit strictly on `train.csv`. | **PASS** |
| **5. Out-of-Order Temporal Splitting** | Random shuffle leaking future patterns | Strict chronological splitting ($2023 \rightarrow 2024\text{H1} \rightarrow 2024\text{H2}$). | **PASS** |

---

## 3. Classification of Feature Availability

The feature availability audit ([`ml/data/modeling/v1/feature_availability.csv`](file:///home/suraj/atmosIQ/ml/data/modeling/v1/feature_availability.csv)) classifies all 255 features into:
- **`SAFE_HISTORICAL_FEATURE`**: 183 features (`*_lag_*d`, `*_roll_*d`). `prediction_safe: true`.
- **`STATIC_CALENDAR_FEATURE`**: 17 features (`day_of_week`, `month`, `is_weekend`, `is_stubble_season`, etc.). `prediction_safe: true`.
- **`SAME_DAY_FEATURE`**: 54 features (same-day weather/fire/chemical metrics measured on day $t$). `prediction_safe: false` for strict day $t-1$ cutoff.
- **`TARGET_VARIABLE`**: 1 feature (`pm25`).

---

## 4. Final Conclusion

All 200 safe predictors satisfy temporal availability requirements.

Final Audit Result: **PASS**
