# AtmosIQ Production Uncertainty Layer (v1.0.0)

## Architecture Overview
The AtmosIQ Uncertainty Layer is strictly decoupled from the point forecasting model.

- **Point Forecasting Model**: `MODEL_V3_PRODUCTION` (`ml/models/production/v3/model.joblib`)
- **Uncertainty Calibration Method**: `normalized_conformal` (Normalized Heteroscedastic Conformal Prediction)
- **Feature Registry**: Exactly 35 prediction-safe features

## Validated Performance Metrics (2022–2024, N = 1,096 Held-Out Observations)
- **80% Empirical Coverage**: `80.66%`
- **90% Empirical Coverage**: `89.78%`
- **95% Empirical Coverage**: `95.71%`
- **Extreme (>=150 µg/m³) Coverage**: `89.45%`
- **Severe (>=250 µg/m³) Coverage**: `89.01%`
- **90% Mean Prediction Interval Width (MPIW)**: `68.77 µg/m³`
- **90% Winkler Interval Score**: `88.22`

## Usage Instructions
Given a new point forecast $\hat{y}$ and predicted pollution regime $r$:
1. Retrieve regime dispersion scale $\sigma_r$ from `calibration_artifacts.json`.
2. Compute adaptive margin: $\Delta_{1-\alpha} = q_{1-\alpha} \cdot (\sigma_r + \epsilon)$.
3. Construct physically bounded prediction interval:
   $$[\max(0, \hat{y} - \Delta_{1-\alpha}), \hat{y} + \Delta_{1-\alpha}]$$
