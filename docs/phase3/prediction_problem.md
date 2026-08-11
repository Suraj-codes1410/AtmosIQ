# AtmosIQ Phase 3A: Prediction Problem Definition

## 1. Primary Objective & Task Definition

AtmosIQ is an **Explainable AI platform for PM2.5 source-signal attribution and policy intelligence**.

The core predictive task defined for Phase 3 models is:

$$\text{Predict ambient PM2.5 concentration for day } t \quad (y_t) \quad \text{using information available BEFORE day } t \quad (X_{t-1 \text{ and earlier}})$$

- **Target Variable ($y_t$)**: Daily average PM2.5 concentration ($\mu\text{g/m}^3$) on day $t$.
- **Information Cutoff**: End of day $t-1$ ($23:59:59 \text{ IST}$).
- **Predictor Set ($X$)**: Historical pollutant concentrations, historical meteorology, satellite active fire hotspot metrics, wind vectors, and deterministic calendar indicators up to day $t-1$.

---

## 2. Temporal Availability & Prediction Horizon

| Attribute | Value / Specification |
|---|---|
| **Target Variable ($y$)** | `pm25` on day $t$ |
| **Prediction Cutoff** | End of day $t-1$ |
| **Temporal Resolution** | Daily (24-hour average) |
| **Prediction Horizon** | 1-day ahead forecast / attribution horizon |
| **Geographic Scope** | National Capital Region (NCR) of Delhi, India |

---

## 3. Allowed vs Forbidden Information

### Allowed Predictors ($X_{t-1}$ & Earlier)
- Lagged PM2.5 values: $\text{PM2.5}_{t-1}, \text{PM2.5}_{t-2}, \dots, \text{PM2.5}_{t-14}$.
- Lagged meteorological variables: Temperature, Relative Humidity, Wind Speed, Wind Vector ($u, v$), Pressure, Rainfall up to $t-1$.
- Historical satellite biomass fire hotspot metrics: Fire count, brightness, transport score up to $t-1$.
- Deterministic calendar indicators: `day_of_week`, `month`, `is_weekend`, `is_holiday`, `is_stubble_season`, `days_until_diwali`.

### Forbidden Information (Future Relative to Cutoff $t-1$)
- Same-day actual PM2.5 measurement $y_t$.
- Same-day actual pollutant measurements $\text{PM10}_t, \text{NO2}_t, \text{SO2}_t, \text{CO}_t, \text{O3}_t$.
- Future meteorological or fire hotspot measurements from days $t+1, t+2, \dots$.
- Unshifted rolling window statistics that include day $t$ measurements.
