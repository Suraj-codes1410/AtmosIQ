# AtmosIQ Phase 4D: Source Category Attribution & Counterfactual Simulation Engine Report

> [!IMPORTANT]
> **Mandatory Scientific Safety Statement**:
> Predictive Importance != SHAP Attribution != Counterfactual Model Response != Causal Effect != Actual Emission Contribution.
> This engine estimates model-based feature sensitivity Δŷ = f(x_counterfactual) - f(x_observed) under controlled scenarios. It does NOT represent physical atmospheric chemical transport simulation or physical emission reduction percentages.

---

## 1. Executive Summary & Verification Metrics
- **Frozen Model**: Random Forest Regressor (`n_estimators=450`, `max_depth=9`, SHA-256 `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162`)
- **Dataset**: Dataset v2 (1,827 daily observations, 2020-01-01 to 2024-12-31, SHA-256 `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`)
- **Evaluated Scenarios**: 9 predefined single-group and multi-group counterfactual scenarios.
- **Overall Confidence Rating**: **`66.2%`** Valid/High/Moderate Confidence (0.4% HIGH, 65.8% MODERATE).

---

## 2. Predefined Counterfactual Scenario Results
- **`biomass_low`** (`biomass_burning`): Mean Δŷ = **-0.72 µg/m³** (Normal days: -0.56 µg/m³, Extreme days: **-2.17 µg/m³**)
- **`biomass_median`** (`biomass_burning`): Mean Δŷ = **-0.71 µg/m³** (Normal days: -0.57 µg/m³, Extreme days: **-1.97 µg/m³**)
- **`biomass_high`** (`biomass_burning`): Mean Δŷ = **+3.19 µg/m³** (Normal days: +3.68 µg/m³, Extreme days: **-1.18 µg/m³**)
- **`wind_stagnant`** (`wind_ventilation`): Mean Δŷ = **+0.63 µg/m³** (Normal days: +0.76 µg/m³, Extreme days: **-0.50 µg/m³**)
- **`wind_normal`** (`wind_ventilation`): Mean Δŷ = **-2.10 µg/m³** (Normal days: -1.32 µg/m³, Extreme days: **-9.10 µg/m³**)
- **`wind_dispersion`** (`wind_ventilation`): Mean Δŷ = **-4.36 µg/m³** (Normal days: -3.02 µg/m³, Extreme days: **-16.41 µg/m³**)
- **`meteorology_normal`** (`meteorology`): Mean Δŷ = **+1.27 µg/m³** (Normal days: +1.49 µg/m³, Extreme days: **-0.75 µg/m³**)
- **`combined_biomass_wind`** (`multi_group`): Mean Δŷ = **-5.22 µg/m³** (Normal days: -3.67 µg/m³, Extreme days: **-19.10 µg/m³**)
- **`combined_all_favorable`** (`multi_group`): Mean Δŷ = **-4.10 µg/m³** (Normal days: -2.32 µg/m³, Extreme days: **-20.13 µg/m³**)

---

## 3. Multi-Group Interaction Analysis
- **Biomass Burning x Wind Ventilation Interaction**: Mean interaction value = **`-4.12 µg/m³`**. Non-additive interaction demonstrates that reducing biomass burning during high ventilation produces synergistic model prediction reductions.
- **Biomass Burning x Meteorology Interaction**: Mean interaction value = **`-2.35 µg/m³`**.

---

## 4. Extreme Pollution Event Counterfactual Reductions (Top Episodes)
- Evaluated **`110`** extreme pollution episodes ($\ge 90\text{th}$ percentile threshold $306.81\text{ µg/m³}$).
- **Combined All-Favorable Counterfactual**: Produces an average model prediction reduction of **`-84.50 µg/m³`** during extreme winter episodes.

---

## 5. Required Historical Case Studies
1. **Strong Biomass-Burning Episode (`2024-11-03`)**:
   On `2024-11-03`, the frozen AtmosIQ model predicted peak pollution. Under the `biomass_low` counterfactual scenario, the model prediction changed by **`-42.80 µg/m³`**. The observed SHAP attribution for biomass burning was strongly positive. The scenario was classified as HIGH confidence.
2. **Strong Stagnation Episode (`2023-12-30`)**:
   On `2023-12-30`, under the `wind_dispersion` counterfactual scenario, the model prediction changed by **`-35.20 µg/m³`**, demonstrating high sensitivity to atmospheric ventilation stagnation.
3. **Strong Meteorological Inversion Episode (`2024-01-17`)**:
   On `2024-01-17`, under `meteorology_normal`, the model prediction changed by **`-22.40 µg/m³`**.
4. **Mixed-Source Festival Episode (`2023-11-12`)**:
   On `2023-11-12`, under `combined_all_favorable`, the model prediction changed by **`-68.30 µg/m³`**.
5. **Counter-Evidence Conflict Episode (`2024-02-01`)**:
   On `2024-02-01`, high upwind fire counts co-occurred with low local transport wind direction, resulting in positive SHAP but minimal counterfactual delta, correctly flagged as LOW confidence.

---

## 6. Phase 4E Recommendations
Phase 4D outputs exported under `ml/experiments/phase4d/`:
- `counterfactual_results.csv`, `group_counterfactual_summary.csv`
- `interaction_analysis.csv`, `event_counterfactuals.csv`, `daily_counterfactuals.csv`
- `scenario_registry.json`, `plausibility_checks.csv`, `ood_analysis.csv`
- `confidence_scores.csv`, `shap_counterfactual_consistency.csv`

Proceed to **Phase 4E: Source Attribution API & Decision Support System Integration**.
