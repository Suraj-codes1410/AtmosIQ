# AtmosIQ Production Decision-Support Layer (v1.0.0)

This directory contains the production uncertainty-aware decision support package for AtmosIQ.

## Architecture
- **Forecasting Model**: `MODEL_V3_PRODUCTION` (`v3.0.0-frozen`, 35 features).
- **Uncertainty Engine**: `normalized_conformal` (`v1.0.0`, heteroscedastic conformal prediction).
- **Attribution Engine**: TreeSHAP with 6 environmental process groups.
- **Counterfactual Engine**: 8 validated policy intervention scenarios with directional stability metadata.
- **OOD Gating**: Standardized feature distance scaling.
- **Decision Engine**: Deterministic 3-tier reliability classification.

## Production Artifacts
- `decision_support_schema.json`: Canonical machine-readable JSON schema.
- `decision_rules.json`: Deterministic tier rules and thresholds.
- `method_registry.json`: Production uncertainty and decision-support registry.
- `integration_metadata.json`: Provenance and cryptographic hashes.
- `validation_summary.json`: Multi-year walk-forward validation metrics.
