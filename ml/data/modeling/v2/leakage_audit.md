# Dataset v2 Leakage Audit Report

**Status**: **PASS**  
**Audit Timestamp**: 2026-08-13 01:51:59

1. **Target Leakage**: `pm25` is strictly excluded from predictor matrices $X$.
2. **Lag / Rolling Protection**: All target-derived rolling statistics shifted by $\ge 1$ day.
3. **Temporal Partitioning**: Strict chronological splits (Train 2020-2022, Val 2023, Test 2024). Zero overlap.
