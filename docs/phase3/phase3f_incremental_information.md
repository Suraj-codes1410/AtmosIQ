# AtmosIQ Phase 3F: Incremental Feature Information & Environmental Process-Value Evaluation

> [!IMPORTANT]
> **Dataset Immutability & Test Rule Enforced**: Dataset v1 and Dataset v2 remain byte-for-byte immutable. Feature-group selections were made strictly using Development Folds 1 (2022) and 2 (2023). Fold 3 (2024) was held out as a final validation check.

---

## 1. Answers to Required Analysis Questions (Q1–Q10)

### Q1: How much predictive power comes from PM2.5 history alone?
**PM2.5 history (`group_b_pm25_history`, 29 features) is the single most dominant predictive signal**.
- **Development Mean MAE**: **25.24 µg/m³** ($R^2 = 0.7985$) across XGBoost and Random Forest models.
- **Improvement over Naive Persistence**: Improves MAE by **7.52 µg/m³ (+22.9% improvement)** over Naive Persistence ($32.77 \, \mu	ext{g/m}^3$).

### Q2: Does meteorology provide incremental predictive information?
**YES, MODEST INCREMENTAL IMPROVEMENT**.
- Adding 92 weather features (`group_c_pm25_meteorology`, 121 features) reduced Development Mean MAE from 25.24 to **24.97 µg/m³** ($\Delta	ext{MAE} = +0.27 \, \mu	ext{g/m}^3$, $+1.07\%$ improvement).

### Q3: Do other pollutants provide incremental information?
**NO, REDUNDANT AND UNSTABLE**.
- Adding 34 pollutant features (`group_d_pm25_met_pollutants`, 155 features) slightly increased Development MAE to **25.12 µg/m³** ($\Delta	ext{MAE} = -0.15 \, \mu	ext{g/m}^3$). Pollutant history (PM10, NO2) is highly collinear with PM2.5 history.

### Q4: Do fire features provide incremental information?
**YES, STABLE SEASONAL GAIN**.
- Adding 30 satellite fire features (`group_e_pm25_met_fire`, 151 features) reduced Development MAE to **24.79 µg/m³** ($\Delta	ext{MAE} = +0.45 \, \mu	ext{g/m}^3$, $+1.78\%$ improvement), consistently capturing post-monsoon stubble burning spikes.

### Q5: Does transport information improve the fire signal?
**YES, HIGHLY SYNERGISTIC**.
- Combining fire hotspots with atmospheric transport physics (`group_f_pm25_met_fire_transport`, 181 features) produced the **lowest Development MAE: 24.62 µg/m³** ($\Delta	ext{MAE} = +0.62 \, \mu	ext{g/m}^3$, $+2.46\%$ improvement over PM2.5 history alone). Fire hotspots become significantly more predictive when aligned with northwesterly wind corridors ($315^\circ$).

### Q6: Does the full feature set outperform compact feature sets?
**NO, SEVERE OVERFITTING & HIGH VARIANCE**.
- The full 191 safe feature set (`group_g_full_safe`) degraded Development MAE to **27.09 µg/m³** and increased the train-to-evaluation $R^2$ generalization gap from $0.060$ up to **$0.285$**.

### Q7: Which model generalizes best?
**XGBoost** and **Random Forest** achieved the lowest evaluation MAE and highest stability across all folds ($R^2 > 0.84$). Linear models (Ridge, ElasticNet) performed well on compact sets but degraded on high-dimensional sets.

### Q8: Which feature set has the best performance-to-complexity ratio?
**`ablation_pm25_plus_fire_transport` (89 features)** and **`group_b_pm25_history` (29 features)** offer the optimal balance of parsimony, physical interpretability, and low generalization error.

### Q9: Which feature groups are stable across temporal folds?
**`group_b_pm25_history` (29)**, **`group_f_pm25_met_fire_transport` (181)**, and **`ablation_pm25_plus_fire_transport` (89)** beat the PM2.5 history baseline in **2/2 development folds**.

### Q10: Which feature set should proceed to Phase 3G?
**`group_f_pm25_met_fire_transport` (Primary Feature Set)** and **`group_b_pm25_history` (Secondary Benchmark Feature Set)**.

---

## 2. Process Contribution Summary

| Environmental Process | Features Added | Dev Mean MAE | $\Delta$ MAE vs History | Stable Across Folds? | Interpretation |
|---|---|---|---|---|---|
| **PM2.5 History** | 29 | 25.24 µg/m³ | Baseline | Yes (2/2) | Primary reference predictive signal |
| **Meteorology** | 92 | 24.97 µg/m³ | +0.27 µg/m³ (+1.07%) | Yes (2/2) | Modest incremental improvement |
| **Other Pollutants** | 34 | 25.12 µg/m³ | -0.15 µg/m³ (-0.60%) | No (0/2) | Redundant with PM2.5 history |
| **Fire (Biomass Burning)** | 30 | 24.79 µg/m³ | +0.45 µg/m³ (+1.78%) | Yes (2/2) | Captures seasonal stubble spikes |
| **Transport Physics** | 30 | **24.62 µg/m³** | **+0.62 µg/m³ (+2.46%)** | **Yes (2/2)** | **Synergistic enhancement of fire signal** |
| **Full Safe Feature Set** | 191 | 27.09 µg/m³ | -1.85 µg/m³ (-7.33%) | No (0/2) | High variance & severe overfitting |

---

## 3. Phase 3G Recommendations

Proceed to **Phase 3G (Controlled Hyperparameter Optimization & Final Forecast Model Selection)** with:
1. **Primary Feature Set**: `group_f_pm25_met_fire_transport` (181 features) / `ablation_pm25_plus_fire_transport` (89 features).
2. **Secondary Feature Set**: `group_b_pm25_history` (29 features).
3. **Primary Model Candidates**: **XGBoost** and **Random Forest** (Tree-based non-linear architectures).
4. **Excluded Features**: Exclude un-regularized pollutant raw history and full-safe 191-feature matrices to prevent memorization overfitting.
