# AtmosIQ Frozen Production Model (v3)

## Overview
This directory contains the authoritative, frozen production model for the AtmosIQ Delhi NCR PM2.5 forecasting platform.

- **Model Version**: `MODEL_V3_PRODUCTION` (`v3.0.0-frozen`)
- **Model Type**: Random Forest Regressor (`scikit-learn` 1.9.0)
- **Feature Set**: `Candidate_C_V3_Compact` (35 prediction-safe features)
- **Model SHA-256**: `9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210`
- **Dataset SHA-256**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736`
- **Performance**: 3-Fold Walk-Forward MAE = 17.0158 µg/m³, R² = 0.9497

## Scientific Disclaimer
> **PREDICTIVE IMPORTANCE ≠ SHAP ATTRIBUTION ≠ COUNTERFACTUAL MODEL RESPONSE ≠ CAUSAL EFFECT ≠ ACTUAL EMISSION CONTRIBUTION**
