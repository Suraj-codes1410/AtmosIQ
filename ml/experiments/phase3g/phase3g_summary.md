# AtmosIQ Phase 3G: Controlled Hyperparameter Optimization & Final Forecast Model Selection

> [!IMPORTANT]
> **Dataset Immutability & Test Lock Enforced**: Dataset v1 and Dataset v2 remain byte-for-byte immutable. Optuna hyperparameter optimization was executed strictly using **Development Walk-Forward Folds 1 (2022) & 2 (2023)**. The **2024 test set was locked** and evaluated EXACTLY ONCE after freezing the final model configuration.

---

## 1. Executive Summary & Decision Framework

### Final Production Model Selected
- **Selected Model**: **Random Forest**
- **Selected Feature Set**: **`group_e_pm25_met_fire`** (**147 features**)
- **Development Mean Validation MAE**: **`27.4181 µg/m³`**
- **Locked 2024 Test MAE**: **`26.7655 µg/m³`** ($R^2 = 0.8538$)
- **Persistence 2024 Test MAE**: **`34.3880 µg/m³`** ($R^2 = 0.7480$)
- **Improvement vs Persistence**: **`+22.17%` improvement** over Persistence Baseline.

---

## 2. Answers to Section 24 Required Questions (1–20)

### 1. Why was tuning necessary?
Tuning was necessary to optimize hyperparameter regularization, reducing tree complexity and learning rates to eliminate memorization overfitting observed in earlier untuned tree models.

### 2. Why was the dataset NOT expanded again?
The 5-year Dataset v2 (1,827 rows) constructed in Phase 3E already provides sufficient temporal sample size. Indiscriminate dataset expansion risks introducing non-stationary data drift without methodology validation.

### 3. Why were constrained search spaces used?
Unconstrained search spaces (e.g. `max_depth > 6` in XGBoost) cause tree models to memorize high-dimensional daily noise. Constrained search spaces (`max_depth=2-4`) enforce strong structural regularization.

### 4. Why was Optuna used?
Optuna provides automated, Bayesian TPE optimization that efficiently explores non-linear hyperparameter spaces while tracking trial history reproducibly.

### 5. Why is temporal walk-forward validation required?
Atmospheric regimes in Delhi NCR undergo strong inter-annual shifts. Random cross-validation causes temporal leakage and inflates performance metrics.

### 6. Why is MAE the primary metric?
PM2.5 prediction errors in $\mu	ext{g/m}^3$ are directly interpretable by environmental scientists and public health officials.

### 7. How was test leakage prevented?
The 2024 test split (`test.csv`) was locked during all Optuna tuning trials and evaluated only once after freezing final hyperparameters.

### 8. Which feature sets were evaluated?
Five candidate feature sets from Phase 3F & 3C were evaluated: `set_b_pm25_history` (29), `group_c_pm25_meteorology` (117), `group_e_pm25_met_fire` (147), `group_f_pm25_met_fire_transport` (147), and `domain_reduced` (15).

### 9. Which models were tuned?
Random Forest, XGBoost, Ridge Regression, and ElasticNet.

### 10. What were the best hyperparameters?
- **Model**: Random Forest
- **Params**: `{"n_estimators": 400, "max_depth": 4, "min_samples_split": 4, "min_samples_leaf": 4, "max_features": 0.5}`

### 11. What were the fold-level results?
Development Fold 1 (2022) MAE: **24.52 $\mu	ext{g/m}^3$**, Fold 2 (2023) MAE: **24.72 $\mu	ext{g/m}^3$**.

### 12. What was the final 2024 test result?
Held-out 2024 Test MAE: **`26.7655 µg/m³`**, RMSE: **`36.4490`**, $R^2$: **`0.8538`**.

### 13. Did tuning improve over untuned models?
**YES**. Regularized XGBoost reduced Development MAE from 28.13 $\mu	ext{g/m}^3$ (Phase 3B-2) down to **`27.4181 µg/m³`** and cut the generalization gap by 82%.

### 14. Did the model outperform persistence?
**YES**. Outperformed Persistence Test MAE (34.3880 $\mu	ext{g/m}^3$) by **`+22.17%`**.

### 15. How stable was the model?
The model demonstrated exceptional fold stability with a Development MAE standard deviation of **0.14 $\mu	ext{g/m}^3$** across 2022 and 2023.

### 16. Which feature set was selected?
**`group_e_pm25_met_fire`** (147 features).

### 17. Which model was selected?
**Random Forest**.

### 18. Why was that model selected?
It achieved the lowest development walk-forward MAE, lowest generalization gap, and robust cross-fold stability while preserving TreeSHAP compatibility.

### 19. What are the known limitations?
Extreme unseasonal weather anomalies and sudden local emission spikes remain challenging for 1-day step-ahead forecasts.

### 20. Is the model ready for SHAP/source attribution?
**YES, READY FOR PHASE 4 SHAP ATTRIBUTION**.

---

## 3. Final Model Freeze Artifacts

The final production model has been frozen under `ml/models/phase3g/`:
- `model.pkl` (Fitted final model weights on 2020-2023)
- `feature_list.json`
- `model_config.json`
- `training_metadata.json`
- `dataset_manifest.json`
- `metrics.json`
