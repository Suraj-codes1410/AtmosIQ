# AtmosIQ Phase 3B-1: Baseline Model Performance & Benchmark Specification

> [!IMPORTANT]
> These models are untuned baselines established to define the performance floor for AtmosIQ. No final model has been selected.

---

## 1. Objective & Scope

The goal of **Phase 3B-1** is to establish transparent, deterministic baseline forecasting performance for Delhi NCR PM2.5 concentrations before introducing non-linear tree-based gradient boosting models (Phase 3B-2).

---

## 2. Dataset & Prediction Cutoff Definition

- **Source of Truth**: Frozen Phase 3A dataset ([`ml/data/modeling/v1/`](file:///home/suraj/atmosIQ/ml/data/modeling/v1/)).
- **Prediction Target ($y_t$)**: Daily mean PM2.5 concentration (`pm25`) on day $t$.
- **Prediction Cutoff**: End of Day $t-1$ ($23:59:59\text{ IST}$).
- **Predictor Set ($X$)**: Features marked `prediction_safe == True` in `feature_availability.csv` (201 features; 184 historical lag/roll + 17 static calendar).
- **Excluded Features**: 53 same-day features measured on day $t$, raw date identifier, and target variable `pm25`.

---

## 3. Methodology & Preprocessing Rules

### A. Preprocessing Pipeline
- **StandardScaler**: Applied to continuous numerical predictor features.
- **Strict Preprocessing Rule**: The scaler is fitted **ONLY on `X_train`** inside an `sklearn.pipeline.Pipeline`. Validation and test splits are transformed using training parameters. Zero validation or test data enters scaler fitting.

### B. Baseline Models Implemented
1. **Persistence Baseline**: $\hat{y}_t = \text{PM2.5}_{t-1}$. For the first day of Validation ($2024-01-01$) and Test ($2024-07-01$), predictions use the observed final day of the preceding period ($2023-12-31$ and $2024-06-30$ respectively).
2. **Linear Regression**: Ordinary Least Squares (`sklearn.linear_model.LinearRegression`).
3. **Ridge Regression**: L2-regularized linear regression (`sklearn.linear_model.Ridge(alpha=1.0)`).

---

## 4. Multi-Period Benchmark Evaluation Results

| Model | Split | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Median AE ($\mu\text{g/m}^3$) |
|---|---|---|---|---|---|
| **Persistence Baseline** | **Train** | 36.5911 | 52.7769 | 0.6867 | 23.1800 |
| **Persistence Baseline** | **Validation** | 31.9925 | 42.2714 | 0.6759 | 26.4200 |
| **Persistence Baseline** | **Test** | 33.5436 | 49.8988 | 0.7894 | 18.9000 |
| **Linear Regression** | **Train** | 17.6777 | 22.9139 | 0.9409 | 13.7442 |
| **Linear Regression** | **Validation** | 96.7369 | 117.5116 | -1.5047 | 83.1788 |
| **Linear Regression** | **Test** | 62.4238 | 84.2014 | 0.4005 | 49.9492 |
| **Ridge Regression ($\alpha=1.0$)** | **Train** | 20.0346 | 26.6943 | 0.9199 | 16.0118 |
| **Ridge Regression ($\alpha=1.0$)** | **Validation** | 59.5240 | 72.9659 | 0.0343 | 55.9771 |
| **Ridge Regression ($\alpha=1.0$)** | **Test** | 48.1660 | 61.9125 | 0.6759 | 40.9942 |

---

## 5. Key Experimental Observations

### A. Overfitting & Multicollinearity
- **Ordinary Least Squares**: Achieves high training fit ($R^2 = 0.9409$), but collapses on Validation ($R^2 = -1.5047$) due to strong multicollinearity among 201 lag and rolling window features.
- **Ridge Regularization**: L2 regularization ($\alpha=1.0$) controls coefficient explosion, recovering test performance to $R^2 = 0.6759$.

### B. Atmospheric Persistence Strength
- Naive Persistence ($\hat{y}_t = \text{PM2.5}_{t-1}$) yields $R^2 = 0.6759$ on Validation and $R^2 = 0.7894$ on Test, proving that daily atmospheric inertia is an essential benchmark.

### C. High-Pollution Evaluation Status
- **Status**: **`HIGH_POLLUTION_THRESHOLD: NOT YET DEFINED`**. No project threshold has been formally defined yet; no arbitrary threshold was fabricated.

---

## 6. Limitations & Next Steps

1. **Linear Model Capacity**: Linear models cannot model complex non-linear interaction effects between meteorological boundary layer heights, humidity, and regional stubble burning transport vectors.
2. **Phase 3B-2**: Proceed to non-linear tree-based models (Random Forest, LightGBM, XGBoost, CatBoost).
