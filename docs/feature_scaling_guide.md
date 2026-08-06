# atmosIQ Feature Scaling & Preprocessing Recommendations (Step 9)

This document provides explicit preprocessing and scaling guidelines for Phase 3 ML model training.

> [!IMPORTANT]
> Feature scaling is NOT applied directly to `feature_dataset.csv` in Phase 2 to preserve raw physical interpretability. These recommendations should be instantiated inside scikit-learn / XGBoost training pipelines during Phase 3.

---

## 1. Features to Keep RAW / Unscaled

These features represent physical binary flags or discrete calendar indicators where scaling destroys semantic interpretation:

- `is_weekend`, `is_holiday`, `is_festival`, `is_stubble_season`
- `is_winter`, `is_summer`, `is_monsoon`, `is_post_monsoon`
- `is_raining`, `festival_window`
- `day_of_week`, `month`, `quarter`

---

## 2. Features Recommended for Robust Standardization (`RobustScaler` / `StandardScaler`)

Features with high variance, skewed distributions, or heavy tail outliers (e.g. fire counts, pollution concentrations, rainfall) should be scaled using `RobustScaler` (which uses median and IQR) to prevent extreme outlier distortion:

- **Pollutants**: `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`
- **Lag Variables**: `pm25_lag_*d`, `pm10_lag_*d`, `no2_lag_*d`, `temperature_c_lag_*d`
- **Rolling Statistics**: `pm25_roll_mean_*d`, `pm25_roll_max_*d`, `fire_hotspot_count_roll_mean_*d`
- **Fire Hotspots**: `fire_hotspot_count`, `fire_hotspot_sum_7d`, `fire_hotspot_sum_14d`, `fire_momentum`, `fire_anomaly`
- **Interaction Terms**: `fire_count_x_wind_speed`, `fire_count_x_wind_dir_nw`, `temp_x_humidity`, `pressure_x_wind`

---

## 3. Features Recommended for Min-Max Normalization (`MinMaxScaler` [0, 1])

Bounded domain indicators that benefit from fixed 0–1 range scaling for neural networks or distance-based algorithms:

- `pm25_pm10_ratio` (Bounded 0 to 1)
- `high_confidence_fire_ratio` (Bounded 0 to 1)
- `co_normalized` (Bounded 0 to 1)
- `humidity_pct` (Bounded 0 to 100 → [0, 1])
- `wind_x`, `wind_y` (Bounded vector components)
- `days_until_diwali`, `days_since_diwali`

---

## 4. Summary Table

| Category | Scaling Method | Rationale |
|---|---|---|
| Binary & Categorical Flags | **None (Raw)** | Preserves exact 0/1 indicator logic |
| Pollutant & Fire Hotspot Metrics | **RobustScaler** | Resilient against extreme winter/stubble outliers |
| Meteorological Continuous Features | **StandardScaler** | Gaussian-like distribution centered at mean 0, variance 1 |
| Chemical & Confidence Ratios | **MinMaxScaler** | Fixed bounding range [0, 1] |
