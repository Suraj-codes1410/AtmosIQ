# atmosIQ Feature Analysis & Environmental Process Breakdown

This document provides a domain science breakdown of every column in the input `master_dataset.csv` and engineered features in `feature_dataset.csv`.

---

## 1. Primary Pollutant Variables

### `pm25` (PM2.5 Concentration - Target Variable)
- **What it represents**: Ambient concentration of fine particulate matter (< 2.5 µm diameter) in µg/m³.
- **Why it matters**: Primary respiratory health hazard in Delhi NCR; target variable for ML regression modeling.
- **Expected Relationship**: Self-autocorrelated across days (`pm25_lag_1d` positive correlation). Highly inversely correlated with wind speed and planetary boundary layer ventilation.
- **Potential Issues**: Strong winter seasonality (Nov-Jan peak), extreme non-linear spikes during Diwali fireworks and stubble burning episodes, outliers exceeding 400 µg/m³.

### `pm10` (PM10 Concentration)
- **What it represents**: Coarse particulate matter (< 10 µm diameter) in µg/m³.
- **Why it matters**: Captures both fine combustion aerosols and coarse road/crustal dust.
- **Expected Relationship**: Highly positively correlated with PM2.5 (`PM2.5 / PM10` ratio indicates combustion vs dust dominance).
- **Potential Issues**: Summer dust storm spikes (May-June) cause non-combustion PM10 surges.

### `no2` (Nitrogen Dioxide)
- **What it represents**: Ambient NO2 gas concentration in µg/m³.
- **Why it matters**: Primary traffic and vehicular combustion proxy.
- **Expected Relationship**: Positive correlation with PM2.5 on weekday morning/evening traffic rush hours.
- **Potential Issues**: Photochemical degradation under intense solar radiation during summer.

### `so2` (Sulfur Dioxide)
- **What it represents**: Ambient SO2 gas concentration in µg/m³.
- **Why it matters**: Industrial coal combustion and thermal power plant emission proxy.
- **Expected Relationship**: Moderate positive baseline correlation with PM2.5 in industrial clusters (Bawana, Narela).

### `co` (Carbon Monoxide)
- **What it represents**: Ambient CO concentration in mg/m³.
- **Why it matters**: Incomplete combustion marker (vehicular exhaust + biomass burning).
- **Expected Relationship**: High positive correlation with PM2.5 during biomass burning and low boundary layer stagnation.

### `o3` (Ground-level Ozone)
- **What it represents**: Photochemical secondary pollutant in µg/m³.
- **Why it matters**: Formed via VOC + NOx reactions driven by solar UV radiation.
- **Expected Relationship**: Anti-correlated with PM2.5 in winter; positively correlated during summer heatwaves.

---

## 2. Meteorological Variables

### `temperature_c` & `temperature_humidity_index`
- **What it represents**: Daily surface mean air temperature (°C) and thermal comfort index.
- **Why it matters**: Temperature inversions in winter trap pollutants near the surface.
- **Expected Relationship**: Inversely correlated with PM2.5 (colder air = lower boundary layer height = higher PM2.5).

### `humidity_pct`
- **What it represents**: Relative humidity (%).
- **Why it matters**: High humidity promotes secondary inorganic aerosol formation (sulfates/nitrates hygro-growth).
- **Expected Relationship**: Positively correlated with PM2.5 during foggy winter mornings.

### `wind_speed_kmh`, `wind_x`, `wind_y`
- **What it represents**: Wind speed at 10m and directional vector components (U, V).
- **Why it matters**: Primary physical dispersion mechanism.
- **Expected Relationship**: Strong inverse relationship with PM2.5 (High wind = rapid advective ventilation).

### `precipitation_mm` & `is_raining`
- **What it represents**: Daily cumulative rainfall (mm).
- **Why it matters**: Wet deposition (rain wash-out effect) scrubs particulates from the atmosphere.
- **Expected Relationship**: Strong negative correlation; rainfall causes sharp immediate drops in PM2.5.

---

## 3. Satellite Biomass Burning Variables

### `fire_hotspot_count`, `fire_hotspot_sum_7d`, `wind_weighted_hotspot_transport_score`
- **What it represents**: MODIS/VIIRS satellite active fire detections in Punjab, Haryana, and Delhi NCR.
- **Why it matters**: Agricultural paddy straw stubble burning in Oct-Nov causes severe hazardous pollution episodes.
- **Expected Relationship**: Strong non-linear positive correlation when combined with Northwest wind vector direction (`wind_weighted_hotspot_transport_score`).
