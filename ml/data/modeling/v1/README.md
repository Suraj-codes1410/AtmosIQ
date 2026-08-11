# AtmosIQ Modeling Dataset v1 (Phase 3A Frozen Snapshot)

## Overview
This directory contains the frozen, versioned, and audit-verified modeling dataset for AtmosIQ Phase 3.

- **Source File**: `ml/data/processed/feature_dataset.csv`
- **Frozen File**: `feature_dataset_frozen.csv`
- **SHA-256 Hash**: `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Total Observations**: 731 daily records (2023-01-01 to 2024-12-31)
- **Target Variable**: `pm25`
- **Prediction Cutoff**: End of Day t-1 ($X_{t-1} ightarrow Y_t$)

---

## Split Structure

1. **Train Set (`train.csv`)**: 2023-01-01 to 2023-12-31 (365 days)
2. **Validation Set (`validation.csv`)**: 2024-01-01 to 2024-06-30 (182 days)
3. **Test Set (`test.csv`)**: 2024-07-01 to 2024-12-31 (184 days)

---

## Artifact Manifests

- `dataset_manifest.json`: Complete feature schema & dataset metadata.
- `split_manifest.json`: Split boundary definitions and cryptographic SHA-256 hashes.
- `feature_availability.csv`: Temporal classification of all features.
- `leakage_audit.md`: Formal leakage prevention verification report (Result: PASS).
- `plots/pm25_temporal_split.png`: Chronological split timeline visualization.
