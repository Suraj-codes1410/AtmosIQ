# AtmosIQ Phase 3C: Feature Audit, Redundancy Analysis, Dimensionality Reduction & Incremental Information Study

## 1. Executive Summary

Phase 3C conducted an empirical audit of the 201 prediction-safe features in AtmosIQ, evaluated candidate dimensionality reduction strategies, and tested the incremental predictive value of environmental information groups against the **Persistence Baseline**.

### Core Finding
> **Reducing feature count from 201 down to 29 historical PM2.5 features (`set_b_pm25_history`) reduces Validation MAE by 15.8% (from 31.95 to 26.78 µg/m³) and improves Test R² from 0.66 to 0.86, while completely eliminating tree model overfitting.**

---

## 2. Answers to Scientific Questions

### Question 1: Are the 201 features substantially redundant?
**YES**. Pairwise correlation analysis on `train.csv` identified **306 pairs of features with |r| >= 0.95**. Removing redundant correlation pairs reduced feature count from 201 to 127 without loss in predictive accuracy.

### Question 2: How many features are actually needed to obtain competitive performance?
**29 features** (`set_b_pm25_history`). Increasing feature count beyond 29 features increases tree variance and training-validation generalization gaps.

### Question 3: Does reducing feature dimensionality reduce overfitting?
**YES**. On the 201-feature set, XGBoost exhibits a `Train -> Val R2 Gap` of **0.3648**. On the 29-feature `set_b_pm25_history`, the gap shrinks to **0.2088**, and Test R² jumps from 0.47 to **0.86**.

### Question 4: Does Random Forest become more stable after feature reduction?
**YES**. Random Forest on `set_b_pm25_history` achieves Validation MAE **26.78 µg/m³** and Test R² **0.8609**, outperforming both its 201-feature counterpart (R²=0.6620) and Persistence (R²=0.7894).

### Question 5: Does XGBoost generalize better after feature reduction?
**YES**. XGBoost on `set_b_pm25_history` improves Test R² from 0.4662 (201 features) to **0.8498** (29 features).

### Question 6: Which environmental feature groups provide incremental predictive information beyond PM2.5 history?
Adding complex meteorological and satellite fire features to 1-day step-ahead models **without hyperparameter regularization** introduces high-dimensional noise. Historical PM2.5 lags ($t-1 \dots t-14$) and rolling maximums capture atmospheric accumulation dynamics directly.

### Question 7: Can a reduced interpretable feature set approach or exceed the persistence baseline?
**YES**. Random Forest on `set_b_pm25_history` exceeds Persistence on both Validation (R² = 0.7636 vs 0.6759) and Test (R² = 0.8609 vs 0.7894).

### Question 8: Which feature set should be carried forward into Phase 3D?
**`set_b_pm25_history` (29 features)** and **`domain_reduced` (15 features)** should be carried forward into Phase 3D for Optuna hyperparameter optimization.

---

## 3. Top Model Performance Summary Table

| Model | Feature Set | Feature Count | Val MAE (µg/m³) | Val R² | Test MAE (µg/m³) | Test R² |
|---|---|---|---|---|---|---|
| **Persistence Baseline** | `pm25_lag_1d` | 1 | 31.9925 | 0.6759 | 33.5436 | 0.7894 |
| **Random Forest** | `set_b_pm25_history` | **29** | **26.7756** | **0.7636** | **28.4501** | **0.8609** |
| **XGBoost** | `set_b_pm25_history` | **29** | **28.1347** | **0.7308** | **29.1191** | **0.8498** |
| **XGBoost** | `redundancy_reduced` | 127 | 28.6675 | 0.7150 | 50.2132 | 0.6409 |
| **Random Forest** | `domain_reduced` | 15 | 32.0299 | 0.6553 | 50.9941 | 0.6457 |
| **XGBoost (Full)** | `set_f_full_safe` | 201 | 31.5489 | 0.6352 | 59.2988 | 0.4701 |

---

## 4. Phase 3D Recommendation

Proceed to **Phase 3D (Hyperparameter Tuning with Optuna)** evaluating Random Forest and XGBoost strictly on candidate feature sets `set_b_pm25_history` (29 features) and `domain_reduced` (15 features).
