# AtmosIQ Phase 8C: Production Synthetic Training Dataset Release

## Release Metadata
- **Corpus Name**: `AtmosIQ_Synthetic_Production`
- **Release Version**: `v1.0.0`
- **Corpus SHA-256**: `8ce3a8c0c6fd0049dd174a0e34b8612077fe5d8d9ee1e6c1eb9156b5fa78ae0e`
- **Total Trajectories**: `3305`
- **Total Observations**: `67838`
- **Feature Count**: `35 (Exact match to feature_registry.csv)`

## Package Structure
```
phase8c_release/
├── synthetic_dataset/
│   ├── synthetic_production_corpus_v1_0_0.parquet
│   └── synthetic_production_corpus_v1_0_0.csv
├── manifests/
│   ├── phase8c_dataset_manifest.json
│   ├── synthetic_provenance_manifest.csv
│   └── synthetic_augmentation_policy.json
├── audits/
│   ├── phase8c_integrity_audit.csv
│   ├── phase8c_data_isolation_audit.csv
│   ├── phase8c_reproducibility.csv
│   └── extreme_tail_governance.csv
├── contracts/
│   └── phase9_training_contract.json
├── hashes/
│   ├── protected_artifacts_pre_sha256.json
│   └── protected_artifacts_post_sha256.json
└── README.md
```

## Mandatory Augmentation Policy
- **Recommended Production Augmentation**: **`25%`** (`RECOMMENDED_PRODUCTION`)
- **Controlled Upper Bound**: **`50%`** (`CONTROLLED_UPPER_BOUND`)
- **Prohibited / Non-Production**: **`100%`** (`NOT_RECOMMENDED`)

## Scientific Language Safeguards
> **`SYNTHETIC DATA != OBSERVED DATA`**  
> **`PHYSICS-INFORMED != PHYSICALLY EXACT`**  
> **`STATISTICAL FIDELITY != CAUSAL VALIDATION`**  
> **`ML UTILITY != SCIENTIFIC TRUTH`**  
> **`SYNTHETIC AUGMENTATION != REAL-WORLD OBSERVATION`**  
