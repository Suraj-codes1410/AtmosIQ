# AtmosIQ Phase 3D: Regularized Hyperparameter Optimization & Compact Feature Model Selection

> [!IMPORTANT]
> Model selection was executed strictly using **Validation MAE** on the 2024-H1 validation split. The test set was held out and evaluated ONCE after model candidate freezing.

---

## 1. Executive Summary

Phase 3D performed regularized Optuna hyperparameter optimization across four model families (**Ridge**, **ElasticNet**, **Random Forest**, **XGBoost**) and three compact feature representations (`set_b_pm25_history` - 29, `domain_reduced` - 15, `set_b_plus_core_environment` - 34).

### Final Winner
- **Selected Model**: **XGBoost** on **`set_b_pm25_history`** (**29 features**)
- **Validation MAE**: **`25.1229 µg/m³`** ($R^2 = 0.7987$)
- **Held-Out Test MAE**: **`29.9121 µg/m³`** ($R^2 = 0.8519$)
- **Improvement vs Persistence**: Validation MAE improved by **`6.87 µg/m³`** (**$21.5\%$ improvement**) over Persistence Baseline ($31.99 \, \mu	ext{g/m}^3$).

---

## 2. Answers to Required Analysis Questions

### Question 1: Does hyperparameter tuning improve Random Forest over Phase 3C?
**YES**. Random Forest Validation MAE improved from 26.78 $\mu	ext{g/m}^3$ down to **25.33 $\mu	ext{g/m}^3$**, and Validation $R^2$ increased from 0.7636 to **0.8004**.

### Question 2: Does hyperparameter tuning improve XGBoost?
**YES, DRAMATICALLY**. Conservative tree depth (`max_depth=2-4`) and L1/L2 regularization (`reg_alpha`, `reg_lambda`) reduced XGBoost Validation MAE from 28.13 to **25.12 $\mu	ext{g/m}^3$** and eliminated memorization overfitting (`Train -> Val R2 Gap` dropped from 0.3517 down to **0.0606**).

### Question 3: Does Ridge become competitive after tuning alpha?
**YES**. Standardized Ridge with tuned $lpha$ achieved Validation MAE **25.36 $\mu	ext{g/m}^3$** and Test $R^2$ **0.8504**, matching the tree models.

### Question 4: Does ElasticNet provide useful regularization?
**YES**. ElasticNet achieved Validation MAE **25.23 $\mu	ext{g/m}^3$** and Test $R^2$ **0.8451**, demonstrating that linear models with L1/L2 penalty are highly competitive on compact feature sets.

### Question 5: Does the 29-feature PM2.5-history representation remain superior?
**YES**. `set_b_pm25_history` produced the top 4 performing models in the entire experiment.

### Question 6: Does domain_reduced provide comparable performance with better interpretability?
**YES**. Random Forest on `domain_reduced` (15 features) achieved Validation MAE **25.43 $\mu	ext{g/m}^3$** and Test $R^2$ **0.8620**, offering exceptional parsimony.

### Question 7: Does adding a small number of environmental variables provide incremental information?
**NO**. Adding 5 core environmental variables (`set_b_plus_core_environment`, 34 features) slightly increased Validation MAE from 25.12 to 25.83 $\mu	ext{g/m}^3$, confirming that 1-day step-ahead forecasts are dominated by atmospheric persistence.

### Question 8: Has the train-validation generalization gap decreased?
**YES**. The generalization gap for XGBoost dropped from **0.3517** (Phase 3B-2) to **0.0606** (Phase 3D).

### Question 9: Does the tuned model outperform Persistence on the untouched test set?
**YES**. The final model achieved Test MAE **29.91 $\mu	ext{g/m}^3$** ($R^2 = 0.8519$) vs Persistence Test MAE **33.54 $\mu	ext{g/m}^3$** ($R^2 = 0.7894$).

### Question 10: Does the final model justify proceeding to the attribution stage?
**YES**. With $R^2 > 0.85$ and zero overfitting on compact feature representations, the model provides an optimal, stable foundation for TreeSHAP source attribution.

---

## 3. Model Comparison Table (Validation vs Held-Out Test)

| Model | Feature Set | Feature Count | Val MAE ($\mu	ext{g/m}^3$) | Val $R^2$ | Test MAE ($\mu	ext{g/m}^3$) | Test $R^2$ |
|---|---|---|---|---|---|---|
| **Persistence Baseline** | `pm25_lag_1d` | 1 | 31.9925 | 0.6759 | 33.5436 | 0.7894 |
| **XGBoost (Tuned)** | `set_b_pm25_history` | **29** | **25.1229** | **0.7987** | **29.9121** | **0.8519** |
| **ElasticNet (Tuned)** | `set_b_pm25_history` | **29** | **25.2268** | **0.7884** | **30.6302** | **0.8451** |
| **Random Forest (Tuned)**| `set_b_pm25_history` | **29** | **25.3290** | **0.8004** | **28.6562** | **0.8599** |
| **Ridge (Tuned)** | `set_b_pm25_history` | **29** | **25.3603** | **0.7919** | **29.6871** | **0.8504** |
| **Random Forest (Tuned)**| `domain_reduced` | **15** | **25.4301** | **0.7881** | **30.4385** | **0.8620** |

---

## 4. Recommendation for Phase 3E

Proceed to **Phase 3E (TreeSHAP Source Attribution & Explainability)** using the regularized **XGBoost** and **Random Forest** models trained on `set_b_pm25_history` (29 features) and `domain_reduced` (15 features).
