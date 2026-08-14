# AtmosIQ Phase 4H: Dataset v3 Production Candidate Evaluation & Model Selection

## 1. Context & Objective
The objective of Phase 4H is to determine rigorously whether Dataset v3 and a v3-trained candidate model provide sufficient, reproducible, and statistically defensible improvement to justify replacing the existing frozen Phase 3G production model (`MODEL_V2_PRODUCTION_CONTROL`).

## 2. Lineage & Provenance
- **Dataset v1**: `c271bfc6df5dc442b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Dataset v2**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`
- **Dataset v3**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736`
- **Frozen Control Model SHA-256**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162`

## 3. Walk-Forward Evaluation Methodology
Chronological expanding window validation across 3 folds:
- **Fold 1**: Train 2020–2021, Test 2022
- **Fold 2**: Train 2020–2022, Test 2023
- **Fold 3**: Train 2020–2023, Test 2024

## 4. Master Model Comparison Results
| model_name   | dataset_version   | feature_set             |   num_features |   mean_mae |   mean_rmse |   mean_r2 |   mean_median_ae |   mean_generalization_gap |   delta_mae_vs_v2 |   delta_rmse_vs_v2 |   delta_r2_vs_v2 |   delta_median_ae_vs_v2 |
|:-------------|:------------------|:------------------------|---------------:|-----------:|------------:|----------:|-----------------:|--------------------------:|------------------:|-------------------:|-----------------:|------------------------:|
| RandomForest | v3                | Candidate_C_V3_Compact  |             35 |    17.0168 |     21.8468 |    0.9494 |          13.781  |                    0.0308 |           -8.6418 |           -12.8572 |           0.0767 |                 -4.4985 |
| RandomForest | v3                | Candidate_B_V3_Expanded |            161 |    17.034  |     21.8397 |    0.9495 |          14.0633 |                    0.034  |           -8.6246 |           -12.8644 |           0.0769 |                 -4.2162 |
| XGBoost      | v3                | Candidate_C_V3_Compact  |             35 |    17.0643 |     22.1425 |    0.9478 |          13.4639 |                    0.0375 |           -8.5943 |           -12.5616 |           0.0752 |                 -4.8156 |
| XGBoost      | v3                | Candidate_B_V3_Expanded |            161 |    17.426  |     22.6879 |    0.9455 |          14.0367 |                    0.0436 |           -8.2326 |           -12.0161 |           0.0728 |                 -4.2427 |
| ElasticNet   | v3                | Candidate_C_V3_Compact  |             35 |    19.272  |     24.3352 |    0.9372 |          16.2853 |                    0.0047 |           -6.3866 |           -10.3688 |           0.0645 |                 -1.9942 |
| Ridge        | v3                | Candidate_C_V3_Compact  |             35 |    19.2929 |     24.377  |    0.9369 |          16.5041 |                    0.0052 |           -6.3657 |           -10.3271 |           0.0643 |                 -1.7754 |
| ElasticNet   | v3                | Candidate_B_V3_Expanded |            161 |    19.6051 |     24.7268 |    0.9351 |          16.7211 |                    0.0096 |           -6.0535 |            -9.9773 |           0.0625 |                 -1.5584 |
| Ridge        | v3                | Candidate_B_V3_Expanded |            161 |    19.9309 |     25.0379 |    0.9335 |          16.7645 |                    0.0131 |           -5.7278 |            -9.6662 |           0.0608 |                 -1.515  |
| Frozen_RF_v2 | v2                | Candidate_A_V2_Baseline |            147 |    25.6586 |     34.704  |    0.8727 |          18.2795 |                    0      |            0      |             0      |           0      |                  0      |
| RandomForest | v3                | Candidate_A_V2_Baseline |            147 |    27.3689 |     37.5278 |    0.8513 |          19.6933 |                    0.0801 |            1.7103 |             2.8237 |          -0.0214 |                  1.4138 |
| XGBoost      | v3                | Candidate_A_V2_Baseline |            147 |    27.7132 |     38.1189 |    0.847  |          20.3364 |                    0.0549 |            2.0546 |             3.4148 |          -0.0256 |                  2.0569 |
| ElasticNet   | v3                | Candidate_A_V2_Baseline |            147 |    28.2594 |     38.4377 |    0.8443 |          20.3479 |                    0.0172 |            2.6007 |             3.7336 |          -0.0284 |                  2.0684 |
| Ridge        | v3                | Candidate_A_V2_Baseline |            147 |    28.5774 |     38.6676 |    0.8424 |          20.5905 |                    0.0234 |            2.9188 |             3.9636 |          -0.0303 |                  2.311  |

## 5. Statistical Significance & Bootstrap Analysis
- **Selected Best Candidate**: `RandomForest` (Candidate_C_V3_Compact)
- **Wilcoxon Signed-Rank Test p-value**: `3.5567e-33`
- **95% Bootstrap Confidence Interval for ΔMAE**: `[-9.7750, -7.2943] µg/m³`
- **Statistically Significant Error Reduction**: `True`

## 6. External Environmental Feature Ablation Study
| ablation_config              | model_name   |   num_features |   mean_mae |   mean_rmse |   mean_r2 |   delta_mae_vs_v2_only |   delta_rmse_vs_v2_only |   delta_r2_vs_v2_only |
|:-----------------------------|:-------------|---------------:|-----------:|------------:|----------:|-----------------------:|------------------------:|----------------------:|
| Model_A_v2_only              | RandomForest |            147 |    27.4428 |     37.5968 |    0.8509 |                 0      |                  0      |                0      |
| Model_B_v2_plus_rainfall     | RandomForest |            152 |    27.36   |     37.4762 |    0.8518 |                -0.0828 |                 -0.1206 |                0.0009 |
| Model_C_v2_plus_pbl          | RandomForest |            151 |    27.4082 |     37.1463 |    0.8541 |                -0.0346 |                 -0.4505 |                0.0032 |
| Model_D_v2_plus_all_external | RandomForest |            161 |    17.0031 |     21.8019 |    0.9497 |               -10.4397 |                -15.7949 |                0.0989 |

## 7. Production Promotion Decision
**DECISION**: `V3 PROMOTION RECOMMENDED`

**Summary**:
Candidate model 'RandomForest' trained on Dataset v3 (Candidate_C_V3_Compact) meets all pre-defined promotion criteria. It achieves a statistically significant overall MAE reduction of 8.6418 ug/m3 (p=3.5567e-33, 95% CI=[-9.7750, -7.2943]) and improves R2 by +0.0767 without extreme-event degradation.

## 8. Reproducibility Information
All experiment logs, metrics, figures, Optuna trial histories, and manifests are saved in `ml/experiments/phase4h/`.
