# AtmosIQ Phase 4A Attribution Model Package v1

This package contains the immutable, reproducible Phase 3G production forecasting model serialized for Phase 4 TreeSHAP attribution.

## Contents
- `model.joblib`: Serialized Random Forest Regressor trained on Dataset v2 (2020-01-01 to 2023-12-31).
- `model_manifest.json`: Full model metadata, hyperparameters, feature order, and SHA-256 checksums.
- `feature_registry.csv`: 147 prediction-safe features in exact model feature order.
- `attribution_groups.csv`: Deterministic mapping from model features to environmental process attribution groups.
- `dataset_manifest.json`: Dataset v2 manifest snapshot (SHA-256: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`).
- `environment.json`: Python environment dependency versions.
- `checksums.txt`: SHA-256 checksums for package integrity verification.

## Interface Contract for Phase 4B
Phase 4B will consume:
1. `model.joblib`
2. `feature_registry.csv`
3. `attribution_groups.csv`
4. `ml/data/modeling/v2/feature_dataset_frozen.csv`

Phase 4B TreeSHAP Reconstruction Check:
$$\text{base\_value} + \sum \text{SHAP\_values} \approx \hat{y}_{\text{pred}}$$
