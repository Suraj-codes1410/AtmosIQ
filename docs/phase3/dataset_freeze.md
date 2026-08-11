# AtmosIQ Phase 3A: Dataset Freeze & Immutability Specification

## 1. Overview & Snapshot Verification

To ensure strict scientific reproducibility across all Phase 3 modeling experiments, the feature engineering matrix ([`ml/data/processed/feature_dataset.csv`](file:///home/suraj/atmosIQ/ml/data/processed/feature_dataset.csv)) has been validated and frozen into an immutable modeling snapshot.

- **Modeling Directory**: `ml/data/modeling/v1/`
- **Frozen Dataset**: `ml/data/modeling/v1/feature_dataset_frozen.csv`
- **Dataset Hash File**: `ml/data/modeling/v1/dataset_hash.txt`
- **Dataset Manifest**: `ml/data/modeling/v1/dataset_manifest.json`

---

## 2. Cryptographic Verification & Hash Specification

The frozen dataset is byte-for-byte identical to the verified Phase 2 output.

$$\text{SHA-256 Hash}: \texttt{c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df}$$

```json
{
    "dataset_name": "AtmosIQ PM2.5 Modeling Dataset",
    "dataset_version": "v1",
    "source_file": "ml/data/processed/feature_dataset.csv",
    "frozen_file": "ml/data/modeling/v1/feature_dataset_frozen.csv",
    "row_count": 731,
    "column_count": 256,
    "target_column": "pm25",
    "date_column": "date",
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "sha256": "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
}
```

---

## 3. Dataset Immutability Rules

1. `feature_dataset_frozen.csv` must **never** be edited, re-scaled, imputed, or mutated in-place.
2. All preprocessing operations (such as `StandardScaler` or `MinMaxScaler`) must be learned strictly on `train.csv` during model fitting.
3. Any discrepancy in SHA-256 hash triggers an immediate script validation failure.
