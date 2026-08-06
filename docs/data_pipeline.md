# atmosIQ Phase 1 Data Engineering Specification

## Overview

Phase 1 of **atmosIQ** establishes the Data Engineering ingestion, validation, and feature-merging pipeline. This pipeline collects ambient air quality, satellite active fire hotspots, meteorology, and calendar indicators for Delhi NCR across 2 full years (731 daily records: 2023-01-01 to 2024-12-31).

```mermaid
graph TD
    A[OpenAQ API / CPCB Stations] -->|PM2.5, PM10, NO2, SO2, CO, O3| E[openaq_delhi_raw.csv]
    B[NASA FIRMS MODIS/VIIRS] -->|Latitude, Longitude, Brightness, Date| F[nasa_firms_raw.csv]
    C[Open-Meteo Historical API] -->|Temp, Humidity, Wind, Pressure, Rain| G[open_meteo_raw.csv]
    D[Calendar Engine] -->|Weekend, Holiday, Festival, Stubble Season| H[calendar_raw.csv]
    
    E --> I[Data Validator & Merge Pipeline]
    F --> I
    G --> I
    H --> I
    
    I --> J[processed/master_dataset.csv (731 Rows × 21 Columns)]
```

---

## Data Sources & Variable Definitions

### 1. OpenAQ Air Quality (`ml/data/raw/openaq_delhi_raw.csv`)
- `pm25` (µg/m³): Fine particulate matter (< 2.5 µm).
- `pm10` (µg/m³): Coarse particulate matter (< 10 µm).
- `no2` (µg/m³): Nitrogen dioxide concentration.
- `so2` (µg/m³): Sulfur dioxide concentration.
- `co` (mg/m³): Carbon monoxide.
- `o3` (µg/m³): Ground-level ozone.

### 2. NASA FIRMS Active Fire Hotspots (`ml/data/raw/nasa_firms_raw.csv`)
- Spatial Bounding Box: Lat 28.0–32.5°N, Lon 74.0–78.5°E (Punjab, Haryana, Delhi stubble corridor).
- `fire_hotspot_count`: Total daily satellite active fire counts.
- `mean_fire_brightness`: Mean brightness temperature (Kelvin).
- `high_confidence_fire_count`: Count of detections with confidence >= 75%.

### 3. Open-Meteo Historical Weather (`ml/data/raw/open_meteo_raw.csv`)
- Coordinates: Delhi (28.6139° N, 77.2090° E).
- `temperature_c`: Daily mean surface temperature (°C).
- `humidity_pct`: Daily mean relative humidity (%).
- `wind_speed_kmh`: Maximum wind speed at 10m (km/h).
- `wind_direction_deg`: Dominant wind direction (°).
- `pressure_hpa`: Surface pressure (hPa).
- `precipitation_mm`: Daily cumulative rainfall (mm).

### 4. Calendar & Seasonal Flags (`ml/data/raw/calendar_raw.csv`)
- `is_weekend`: Binary flag (1 for Saturday/Sunday).
- `is_holiday`: Binary flag for Indian national public holidays.
- `is_festival`: Binary flag for high-emission festival days (Diwali, Holi, Dussehra).
- `is_stubble_season`: Binary flag for peak agricultural burning window (Oct 15 - Nov 30).

---

## Data Validation Suite (`ml/src/preprocessing/data_validation.py`)

The pipeline enforces 4 strict validation checks before outputting `master_dataset.csv`:
1. **Timestamp Uniqueness**: Verifies no duplicate `date` primary keys exist.
2. **Missing Values Threshold**: Ensures null percentage per feature remains below 2%.
3. **Physical Range Bounds**: Validates unit consistency (e.g. PM2.5 between 0-1000 µg/m³, Temperature between -15°C and 55°C).
4. **Timezone Normalization**: All timestamps are formatted into ISO 8601 `YYYY-MM-DD` in `Asia/Kolkata` (IST).

---

## Master Dataset Schema (`ml/data/processed/master_dataset.csv`)

- **Granularity**: 1 row = 1 day (731 rows total).
- **Features**: 21 columns (1 timestamp primary key + 20 environmental & calendar variables).
- **Integrity**: 0 missing values (100% complete).

---

## Execution Workflow

Run the entire pipeline end-to-end:

```bash
./scripts/run_data_pipeline.sh
```

Or execute individual ingestion drivers:

```bash
python3 -m ml.src.ingestion.open_meteo_ingestion
python3 -m ml.src.ingestion.nasa_firms_ingestion
python3 -m ml.src.ingestion.openaq_ingestion
python3 -m ml.src.ingestion.calendar_ingestion
python3 -m ml.src.preprocessing.merge_pipeline
```
