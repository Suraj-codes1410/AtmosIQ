# atmosIQ Phase 2: Feature Engineering & Environmental Process Pipeline

## Executive Summary

Phase 2 transforms the raw daily multi-source dataset ([`master_dataset.csv`](file:///home/suraj/atmosIQ/ml/data/processed/master_dataset.csv)) into a production-grade feature matrix ([`feature_dataset.csv`](file:///home/suraj/atmosIQ/ml/data/processed/feature_dataset.csv)) comprising **731 daily rows and 256 engineered features**.

This pipeline mathematically models the physical, chemical, meteorological, and anthropogenic processes governing PM2.5 concentrations in the National Capital Region (NCR) of Delhi while strictly guaranteeing **zero temporal data leakage**.

---

## 1. Input vs Output Data Specification

| Metric | Input (`master_dataset.csv`) | Output (`feature_dataset.csv`) |
|---|---|---|
| **Date Range** | 2023-01-01 to 2024-12-31 (731 days) | 2023-01-01 to 2024-12-31 (731 days) |
| **Column Count** | 21 columns | **256 columns** |
| **Missing Values** | 0 NaNs | **0 NaNs** |
| **Primary Key** | `date` (`YYYY-MM-DD`) | `date` (`YYYY-MM-DD`) |
| **Target Column** | `pm25` (µg/m³) | `pm25` (µg/m³) |

---

## 2. Modular Architecture ([`ml/src/features/`](file:///home/suraj/atmosIQ/ml/src/features/))

```mermaid
graph TD
    A[master_dataset.csv] --> B[Time & Calendar Extractor]
    A --> C[Weather & Wind Vector Extractor]
    A --> D[Satellite Fire Hotspot Extractor]
    A --> E[Pollution & Chemical Ratio Extractor]
    
    B --> F[Interaction Feature Extractor]
    C --> F
    D --> F
    E --> F
    
    F --> G[Historical Lag Generator]
    G --> H[Rolling Statistics Generator]
    H --> I[Leakage Audit & NaN Imputer]
    I --> J[feature_dataset.csv (731 × 256)]
```

### Module Descriptions

1. **`time_features.py` & `calendar_features.py`**:
   - Extracts cyclical temporal indicators (`day_of_week`, `month`, `quarter`, `season`, `is_weekend`).
   - Computes climate regime indicators (`is_winter`, `is_summer`, `is_monsoon`, `is_post_monsoon`).
   - Computes festival proximity countdowns (`days_until_diwali`, `days_since_diwali`, `festival_window`).
   - Constructs traffic proxy indices (`traffic_activity_proxy`).

2. **`weather_features.py`**:
   - Trigonometric wind vector decomposition ($u, v$ components).
   - Thermal comfort index (`temperature_humidity_index`).
   - Precipitation wash-out dynamics (`is_raining`, `consecutive_rain_days`, `dry_day_count`).
   - 24-hour meteorological rates of change (`temperature_change`, `humidity_change`, `pressure_change`, `wind_speed_change`, `wind_direction_change`).

3. **`fire_features.py`**:
   - Multi-day satellite fire hotspot accumulations (`fire_hotspot_sum_7d`, `fire_hotspot_sum_14d`).
   - Fire dynamics (`brightness_trend`, `high_confidence_fire_ratio`, `fire_acceleration`, `fire_momentum`, `fire_anomaly`).
   - Advective smoke transport vector score (`wind_weighted_hotspot_transport_score`).

4. **`pollution_features.py`**:
   - Chemical pollutant ratios (`pm25_pm10_ratio`, `no2_so2_ratio`).
   - Min-Max normalized CO concentration (`co_normalized`).
   - Daily rate of change and 7-day rolling volatility (`pollutant_volatility`).
   - 30-day baseline Z-score anomaly metrics (`pollutant_zscore`, `pollutant_anomaly_score`).

5. **`interaction_features.py`**:
   - Non-linear process cross-terms (`fire_count_x_wind_speed`, `temp_x_humidity`, `pressure_x_wind`, `festival_x_fire_count`, `weekend_x_traffic_proxy`).

6. **`utils.py`**:
   - Reusable lag transformer (`create_lags`) with $k \in \{1, 2, 3, 7, 14\}$ days.
   - Reusable rolling statistics transformer (`create_rolling_stats`) with windows $W \in \{3, 7, 14, 30\}$ days evaluating Mean, Median, Max, Min, Std, and Variance.

7. **`feature_pipeline.py` & `run_feature_pipeline.py`**:
   - High-level pipeline orchestrator executing load, transformation, validation, and export steps.

---

## 3. Atmospheric Science & Process Mathematics

### A. Trigonometric Wind Vector Decomposition
Wind direction is reported in meteorological degrees ($\theta \in [0^\circ, 360^\circ]$) representing the direction wind is blowing **from**. Vector components ($u, v$) in km/h are derived as:

$$u = -\text{wind\_speed} \times \sin\left(\frac{\pi}{180} \theta\right) \quad (\text{Zonal / East-West Vector})$$

$$v = -\text{wind\_speed} \times \cos\left(\frac{\pi}{180} \theta\right) \quad (\text{Meridional / North-South Vector})$$

### B. Advective Stubble Burning Smoke Transport Score
Stubble burning in Punjab/Haryana occurs Northwest of Delhi ($\sim 315^\circ$). Smoke advection towards Delhi depends on fire hotspot intensity, wind speed, and angular alignment ($\theta_{NW} = 315^\circ$):

$$\text{Alignment} = \max\left(0, \, \cos\left(\frac{\pi}{180}(\theta - 315^\circ)\right)\right)$$

$$\text{Transport Score} = \text{fire\_hotspot\_count} \times \text{wind\_speed\_kmh} \times \text{Alignment}$$

### C. Temperature-Humidity Index (THI)
Measures thermal comfort and boundary layer stability:

$$\text{THI} = T - (0.55 - 0.55 \times \text{RH}) \times (T - 14.5)$$

---

## 4. Anti-Leakage Temporal Design Architecture

To ensure models trained in Phase 3 do not suffer from data leakage:
1. **Chronological Sorting**: Datasets are sorted strictly in ascending order by date.
2. **Lag Features**: All lag features shift by $k \ge 1$ days ($X(t - k)$).
3. **Rolling Statistics**: Pollutant rolling windows shift the input series by 1 day before window evaluation ($Y_{\text{shifted}}(t) = Y(t - 1)$), evaluating range $[t - W, t - 1]$. The target $Y(t)$ at time $t$ is strictly excluded from its predictor features.

---

## 5. Execution & Verification Guide

### Run Feature Engineering Pipeline:
```bash
python run_feature_pipeline.py
```

### Run Unit Test Suite:
```bash
./venv/bin/pytest ml/tests/
```

### Inspect Feature Dataset Schema & Values:
```bash
source venv/bin/activate
python -c "import pandas as pd; df=pd.read_csv('ml/data/processed/feature_dataset.csv'); print(df[['date', 'pm25', 'temperature_c', 'wind_speed_kmh', 'fire_hotspot_count', 'wind_x', 'pm25_lag_1d', 'pm25_roll_mean_7d', 'wind_weighted_hotspot_transport_score']].head(10))"
```
