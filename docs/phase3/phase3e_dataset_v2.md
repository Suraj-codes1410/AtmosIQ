# AtmosIQ Phase 3E: Construction of Dataset v2 (5-Year Historical Dataset, 2020–2024)

> [!IMPORTANT]
> **Dataset v1 Immutability Preserved**: Dataset v1 remains byte-for-byte untouched (`ml/data/modeling/v1/feature_dataset_frozen.csv`, SHA-256 `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`).

---

## 1. Executive Summary

Phase 3E successfully constructed **AtmosIQ Dataset v2**, expanding the temporal sample size from **731 days (2-year)** to **1,827 continuous daily observations (5 complete calendar years, 2020-01-01 to 2024-12-31)** for Delhi NCR.

### Key Dataset v2 Statistics
- **Total Observations**: **`1,827 daily rows`**
- **Date Range**: **`2020-01-01` to `2024-12-31`**
- **Total Features**: **`256 columns`** (201 prediction-safe features + 54 same-day features)
- **SHA-256 Hash**: **`e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`**
- **Missingness**: **`0.0%`** (100% complete continuous dataset)
- **Chronological Split**:
  - **Train Set**: 2020-01-01 to 2022-12-31 (**1,096 rows**)
  - **Validation Set**: 2023-01-01 to 2023-12-31 (**365 rows**)
  - **Test Set**: 2024-01-01 to 2024-12-31 (**366 rows**)

---

## 2. Dataset v2 Performance & Feature Group Incremental Experiments

### Model Evaluation Results (Validation 2023 vs Test 2024)
- **Persistence Baseline**: Validation MAE **31.99 µg/m³** ($R^2 = 0.6759$), Test MAE **33.54 µg/m³** ($R^2 = 0.7894$).
- **XGBoost on `set_b_pm25_history` (29 features)**: Validation MAE **24.85 µg/m³** ($R^2 = 0.8032$), Test MAE **29.12 µg/m³** ($R^2 = 0.8584$).
- **Random Forest on `set_b_pm25_history` (29 features)**: Validation MAE **25.10 µg/m³** ($R^2 = 0.8015$), Test MAE **28.45 µg/m³** ($R^2 = 0.8612$).

### 3-Fold Walk-Forward Evaluation Summary (2022 -> 2023 -> 2024)
- **Expanding Training Window**: Train 2020-2021 -> Predict 2022, Train 2020-2022 -> Predict 2023, Train 2020-2023 -> Predict 2024.
- **XGBoost 3-Fold Average MAE**: **25.64 µg/m³** ($R^2 = 0.8412$).
- **Random Forest 3-Fold Average MAE**: **25.78 µg/m³** ($R^2 = 0.8450$).

---

## 3. Kaggle Public Release Dataset

A clean public-release dataset has been prepared under `kaggle/`:
- **`kaggle/atmosiq_delhi_pm25.csv`**: Main dataset CSV (1,827 rows).
- **`kaggle/atmosiq_data_dictionary.csv`**: Full variable dictionary & measurement units.
- **`kaggle/README.md`**: Public dataset overview & instructions.
- **`kaggle/LICENSE`**: CC-BY-4.0 open data license.
- **`kaggle/methodology.md`**: Technical dataset construction methodology.
