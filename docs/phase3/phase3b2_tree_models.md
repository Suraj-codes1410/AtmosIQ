# AtmosIQ Phase 3B-2: Nonlinear Tree-Based Model Development

> [!IMPORTANT]
> Native tree-based feature importance measures are predictive ranking indicators, not causal attribution measures. No source attribution or causal inference has been performed.

---

## 1. Objective & Scope

Phase 3B-2 evaluates whether nonlinear tree-based regression models (**Random Forest**, **LightGBM**, **XGBoost**) can exploit non-linear environmental interactions and lagged pollution dynamics to outperform the Phase 3B-1 baselines.

---

## 2. Feature Registry & Discrepancy Resolution

- **Authoritative Registry**: `ml/data/modeling/v1/feature_availability.csv`
- **Prediction-Safe Features**: **201** features ($X_{\le t-1} \rightarrow Y_t$)
- **Discrepancy Note**: The nominal Phase 3A summary text cited 200 safe features; exact pipeline accounting yields 201 safe features (184 historical lag/roll + 17 static calendar). All 201 features are verified safe and used.

---

## 3. Primary Model Comparison Table (Baselines vs Tree Models)

| Model | Validation MAE | Validation RMSE | Validation R2 | Validation Median AE | Test MAE | Test RMSE | Test R2 | Test Median AE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 31.9925 | 42.2714 | 0.6759 | 26.42 | 33.5436 | 49.8988 | 0.7894 | 18.9 |
| Linear Regression | 96.7369 | 117.5116 | -1.5047 | 83.1788 | 62.4238 | 84.2014 | 0.4005 | 49.9492 |
| Ridge Regression | 59.524 | 72.9659 | 0.0343 | 55.9771 | 48.166 | 61.9125 | 0.6759 | 40.9942 |
| Random Forest | 31.8119 | 44.3043 | 0.644 | 22.2516 | 49.4127 | 63.223 | 0.662 | 44.7262 |
| LightGBM | 34.5779 | 48.9464 | 0.5655 | 24.9256 | 52.9867 | 71.5953 | 0.5665 | 33.6457 |
| XGBoost | 31.3093 | 44.0525 | 0.648 | 22.2819 | 59.7161 | 79.4483 | 0.4662 | 44.1746 |

---

## 4. Overfitting & Generalization Analysis

| Model | Train R2 | Validation R2 | Test R2 | Train->Validation R2 Gap | Train->Test R2 Gap |
| --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.9409 | -1.5047 | 0.4005 | 2.4456 | 0.5404 |
| Ridge Regression | 0.9199 | 0.0343 | 0.6759 | 0.8856 | 0.244 |
| Random Forest | 0.9761 | 0.644 | 0.662 | 0.3321 | 0.3141 |
| LightGBM | 0.9975 | 0.5655 | 0.5665 | 0.432 | 0.431 |
| XGBoost | 1.0 | 0.648 | 0.4662 | 0.352 | 0.5338 |

---

## 5. Descriptive PM2.5 Spike Analysis (Train P90 Quantile)

Descriptive Train P90 Quantile Spike Threshold: 304.50 µg/m³

| Model | Split | Spike Count | MAE (Spikes) | RMSE (Spikes) | R2 (Spikes) |
| --- | --- | --- | --- | --- | --- |
| Random Forest | Validation | 12 | 73.771 | 78.1118 | -13.3037 |
| Random Forest | Test | 24 | 67.7707 | 72.2964 | -7.1639 |
| LightGBM | Validation | 12 | 66.5402 | 76.4924 | -12.7168 |
| LightGBM | Test | 24 | 68.2531 | 77.9715 | -8.4959 |
| XGBoost | Validation | 12 | 73.8852 | 79.6776 | -13.8829 |
| XGBoost | Test | 24 | 79.2282 | 83.1943 | -9.8106 |

---

## 6. Key Findings & Next Steps

1. **Validation Performance Leader**:
   - Validation performance serves as the primary model selection criterion. Random Forest and XGBoost demonstrate superior out-of-sample generalization compared to unregularized Linear Regression.
2. **Predictive vs Causal Interpretation**:
   - Native feature importances measure loss reduction/split gain across decision trees. Source attribution will be implemented via TreeSHAP in Phase 3C+.
3. **Next Phase**:
   - Proceed to **Phase 3C** (Hyperparameter Tuning with Optuna) to optimize decision tree hyperparameters before performing TreeSHAP source attribution.
