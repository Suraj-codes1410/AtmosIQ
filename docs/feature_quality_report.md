# atmosIQ Feature Quality Report (Step 10)

This report details the data type, source, missingness, variance, distribution characteristics, expected predictive usefulness, and leakage risk for key engineered features in `feature_dataset.csv`.

---

## Feature Quality Matrix

| Feature Name | Type | Source | Description | Missing % | Variance | Distribution | Expected Usefulness | Leakage Risk |
|---|---|---|---|---|---|---|---|---|
| `date` | String | Master | ISO Date primary key | 0.0% | N/A | Uniform | High (Ordering) | None |
| `pm25` | Float | OpenAQ | Current day PM2.5 target | 0.0% | High | Right-skewed | Target Variable | Target |
| `pm25_lag_1d` | Float | OpenAQ | 1-day lagged PM2.5 | 0.0% | High | Right-skewed | Very High | None (Strict Shift) |
| `pm25_lag_7d` | Float | OpenAQ | 7-day lagged PM2.5 | 0.0% | High | Right-skewed | High | None (Strict Shift) |
| `pm25_roll_mean_7d` | Float | OpenAQ | 7-day rolling mean PM2.5 | 0.0% | Medium | Right-skewed | Very High | None (Shifted) |
| `pm25_roll_std_7d` | Float | OpenAQ | 7-day rolling std PM2.5 | 0.0% | Medium | Right-skewed | Medium (Volatility) | None (Shifted) |
| `pm25_pm10_ratio` | Float | OpenAQ | Fine vs coarse ratio | 0.0% | Low | Normal [0,1] | High | Low |
| `no2_so2_ratio` | Float | OpenAQ | Traffic vs industrial ratio | 0.0% | Medium | Right-skewed | High | Low |
| `wind_x` | Float | Open-Meteo | East-West wind vector | 0.0% | Medium | Bimodal | Very High | None |
| `wind_y` | Float | Open-Meteo | North-South wind vector | 0.0% | Medium | Bimodal | Very High | None |
| `temperature_humidity_index` | Float | Open-Meteo | Thermal comfort index | 0.0% | High | Bimodal | High | None |
| `is_raining` | Int | Open-Meteo | Binary rain indicator | 0.0% | Low | Bernoulli | High (Scrubbing) | None |
| `consecutive_rain_days` | Int | Open-Meteo | Consecutive rainy days | 0.0% | Low | Right-skewed | High | None |
| `pressure_change` | Float | Open-Meteo | 24h pressure delta | 0.0% | Low | Normal | High (Fronts) | None |
| `fire_hotspot_count` | Int | NASA FIRMS | Daily satellite fire count | 0.0% | High | Heavy-tailed | Very High | None |
| `fire_hotspot_sum_7d` | Float | NASA FIRMS | 7-day cumulative fires | 0.0% | High | Heavy-tailed | Extremely High | None (Shifted) |
| `wind_weighted_hotspot_transport_score` | Float | FIRMS + Meteo | Advective smoke transport | 0.0% | High | Heavy-tailed | Extremely High | None |
| `days_until_diwali` | Int | Calendar | Count down to Diwali | 0.0% | High | Uniform | Very High | None |
| `festival_window` | Int | Calendar | Diwali ± 3 days flag | 0.0% | Low | Bernoulli | High (Spike) | None |
| `is_stubble_season` | Int | Calendar | Stubble burning season | 0.0% | Low | Bernoulli | High | None |
