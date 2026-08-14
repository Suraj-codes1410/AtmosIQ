# AtmosIQ Dataset v3 — 5-Year PM2.5 + External Environmental Observations (2020–2024)

AtmosIQ Dataset v3 expands the 5-year daily Delhi NCR atmospheric benchmark dataset by integrating independently sourced external environmental observations (Precipitation / Rainfall, Planetary Boundary Layer Height, Aerosol Optical Depth, and 850 hPa Transport Wind Components).

## Key Dataset Features
- **Total Observations**: 1,827 daily rows (`2020-01-01` to `2024-12-31`)
- **Spatial Target**: Delhi NCR Centroid (28.6139°N, 77.2090°E) and Regional Bounding Box
- **Features**: 275 columns (147 Dataset v2 prediction-safe baseline features + 14 external environmental features + raw indicators)
- **Primary External Variables**:
  1. `rainfall_1d`, `rainfall_3d`, `rainfall_7d`, `washout_index_3d` (IMD & ERA5 Reanalysis)
  2. `pblh_1d`, `pblh_min_1d`, `ventilation_index_1d` (ECMWF ERA5)
  3. `aod_550_1d`, `aod_roll_mean_3d` (NASA MODIS Aqua/Terra)
  4. `wind_u_component_1d`, `wind_v_component_1d`, `upwind_stubble_quadrant_1d` (ERA5 850 hPa)

## Public Release Lineage
- **Dataset v1**: 2-year daily benchmark (`2020`–`2021`, 731 rows)
- **Dataset v2**: 5-year daily benchmark (`2020`–`2024`, 1,827 rows)
- **Dataset v3**: 5-year daily benchmark + External Environmental Observations (`2020`–`2024`, 1,827 rows)
