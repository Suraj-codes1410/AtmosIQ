# Dataset v3 Construction & Alignment Methodology

1. **Temporal Alignment**: All daily observation records are standardized to midnight UTC / 24-hour Indian Standard Time (IST) averages.
2. **Spatial Scope**: Delhi National Capital Region (NCR) bounding box [28.2°N - 28.9°N, 76.8°E - 77.4°E].
3. **Leakage Prevention**: All predictive inputs are lagged by at least 1 day ($t-1$) or computed over backward rolling windows ($t-1$ through $t-w$). No same-day observations of the target or simultaneous pollutants are exposed as inference inputs.
4. **Model Subsetting**: The full dataset provides 275 columns for broad research; the frozen production forecasting model strictly uses the registered 35 features.
