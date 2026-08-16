# AtmosIQ — Phase 6F Formal Production Freeze Gate
## Immutable Production Baseline & Downstream Isolation Contract

---

## 1. Formal Freeze Declaration

> **`PHASE 6F IS OFFICIALLY FROZEN AND IMMUTABLE.`**  
> As of August 16, 2026, the complete Phase 6F production integration stack (**`ATMOSIQ_DECISION_SUPPORT v1.0.0`**) and all its upstream predictive, conformal uncertainty, and interpretability artifacts are permanently locked.
>
> **No downstream phase (including Phase 7B through Phase 10) is permitted to modify, overwrite, retrain, recalibrate, regenerate, or silently replace any Phase 6F production artifact.**

---

## 2. Freeze Scope & Architecture Lineage

The frozen production stack constitutes the single authoritative operational baseline for AtmosIQ:

```
                            FROZEN PRODUCTION BASELINE (v1.0.0)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Point Forecasting Engine: MODEL_V3_PRODUCTION (RandomForest, 35 features)           │
│    • SHA-256: 9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210         │
│    • Evaluated on N=1,096 held-out days (2022–2024): R² = 0.9497, MAE = 17.02 µg/m³     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Production Uncertainty Layer: normalized_conformal v1.0.0                           │
│    • Location: ml/uncertainty/production/v1/                                           │
│    • 90% Empirical Coverage: 89.78% (MPIW: 68.77 µg/m³, Winkler Score: 88.22)          │
│    • Extreme Episode (≥250 µg/m³) Coverage: 89.01%                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Interpretability & Counterfactual Layer: Phase 6E                                   │
│    • Location: ml/experiments/phase6e/                                                 │
│    • 6 Environmental Process Groups & 8 Predefined Intervention Scenarios              │
│    • OOD Gating: Spearman ρ = +0.7637 (p < 10⁻¹⁵)                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Unified Decision Support Service: ATMOSIQ_DECISION_SUPPORT v1.0.0                   │
│    • Location: ml/decision_support/production/v1/                                      │
│    • Canonical Schema: decision_support_schema.json                                    │
│    • Deterministic 3-Tier Reliability & Evidence/Counter-Evidence Synthesis             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Protected Phase 6F Artifact Catalog & Cryptographic Hashes

Every artifact below has been computed, recorded, and verified in [`ml/experiments/phase6f/phase6f_freeze_manifest.json`](file:///home/suraj/atmosIQ/ml/experiments/phase6f/phase6f_freeze_manifest.json):

| Artifact Path | Category | SHA-256 Cryptographic Hash |
| :--- | :--- | :--- |
| `ml/models/production/v3/model.joblib` | Production Point Model | `9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210` |
| `ml/models/production/v3/feature_registry.csv` | Feature Registry (35) | `b14800a4ee57f0fbc04993a072b9d4007168f0203f95cb6c0e4c81e1ace001bc` |
| `ml/uncertainty/production/v1/calibration_artifacts.json` | 6D Conformal Calibration | `e3ddc417bc09c3f493c6e67b9708438cc543dbb86badd9bd271f93434dbf68c0` |
| `ml/uncertainty/production/v1/calibration_metadata.json` | 6D Calibration Metadata | `42f3733eb3a9da8e0b06e9655e88ee8db1342135dee3df1eec9d9b8a6760d00b` |
| `ml/uncertainty/production/v1/uncertainty_method.json` | 6D Method Registry | `a7256b6ea9d4d4ec7690f1523067d0d19df77597d4b485c349bbfdac01b33333` |
| `ml/uncertainty/production/v1/validation_summary.json` | 6D Validation Summary | `fd8fd32662a8b0f5a779000eb5960b6123b4724cc385fb39d7c20c1213569bb0` |
| `ml/uncertainty/production/v1/README.md` | 6D Documentation | `76defab9a5c92919eae629d58d31291519369011edf8608c48d199666acb94db` |
| `ml/decision_support/production/v1/decision_support_schema.json` | 6F Canonical Schema | `9cf22f3cce428e7711bf4c32aca1fcbee21568a97c8514060421593e46607a05` |
| `ml/decision_support/production/v1/decision_rules.json` | 6F Decision Rules | `453b992081d17ef4adde76f3fa84cc863130be7601b3e216b9ac17116971bda8` |
| `ml/decision_support/production/v1/method_registry.json` | 6F Method Registry | `8e8758be963766bcee3c75284202b49d962794ac732cdad6a4222c8dab65b051` |
| `ml/decision_support/production/v1/integration_metadata.json` | 6F Metadata | `c2f8e841d25bf5b486c35d6abc3568736ac4a1106b4e768170093f785eca5875` |
| `ml/decision_support/production/v1/validation_summary.json` | 6F Validation Summary | `4822fe2d07ea9cdbf4a07f1a1ba0934f7eab7f0e30bae53194d873a9129d740a` |
| `ml/decision_support/production/v1/README.md` | 6F Documentation | `2eaeafc66b43195991742d89eaae945ac88e87f5b57b5f2af9eaf3fa3b058342` |
| `ml/experiments/phase6f/PHASE_6F_COMPLETION_REPORT.md` | 6F Technical Report | `c3f01e461a6572bf93b2f1626ad6de240614a883cfcc5080d661615139219786` |
| `ml/experiments/phase6f/manifest.json` | 6F Run Manifest | `cf987451bdcd28701890bed5baa5aea8bcf59cfbb4606840f7bcc2a8efe21088` |
| `ml/experiments/phase6f/metadata.json` | 6F Run Metadata | `3d25a7ec517e5a0ef446d3f6c66f51954109b15d3f3908b33c8621b7ce7edf5c` |
| `ml/experiments/phase6f/environment.json` | 6F Environment Info | `e4e3b5321c328f5a7dbab034b3516bd885af3113a80855b3b5774d1a47b877d1` |
| `ml/experiments/phase6f/checksums.txt` | 6F Checksum Manifest | `18896dec10d197285c3d8796605b2b5242fa7628883a52266d41bc734649ba77` |
| `ml/data/modeling/v1/feature_dataset_frozen.csv` | Dataset v1 | `c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df` |
| `ml/data/modeling/v2/feature_dataset_frozen.csv` | Dataset v2 | `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301` |
| `ml/data/modeling/v3/feature_dataset_frozen.csv` | Dataset v3 | `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736` |

---

## 4. Functional & Behavioral Freeze Contract

The freeze protects both file integrity and production operational behavior:
1. **Prediction Schema**: Fixed to `decision_support_schema.json`.
2. **Interval Method**: Frozen to `normalized_conformal v1.0.0` with calibrated quantiles $q_{80} = 1.48, q_{90} = 1.96, q_{95} = 2.45$.
3. **Regime Scaling**: Fixed to Low ($9.42\,\mu\text{g/m}^3$), Moderate ($14.85\,\mu\text{g/m}^3$), High ($28.12\,\mu\text{g/m}^3$), Extreme ($44.81\,\mu\text{g/m}^3$).
4. **Decision Rules**: Deterministic thresholds ($\text{relative width} \le 0.65$, $\text{OOD} \le 2.0$, $\text{stability} \ge 0.70$).
5. **Causality Safeguards**: All scientific disclaimers are hardcoded and non-negotiable.

---

## 5. Downstream Isolation & Dependency Direction

```
                            STRICT ONE-WAY DEPENDENCY
           Phase 6F (Frozen Production Baseline)
                           │
                           ▼ (Read-Only)
           Phase 7A (Synthetic Design Spec)
                           │
                           ▼ (Read-Only)
           Phase 7B (Core Generator Execution)
                           │
                           ▼
           Phase 7C / 7D / 7E (Validation & Packaging)
```

- **Rule 1 (Read-Only)**: Downstream phases (7B–10) may READ Phase 6F artifacts to establish comparative baselines or feature registries.
- **Rule 2 (Zero Write)**: Downstream phases are strictly forbidden from writing to, modifying, or regenerating any file in `ml/models/production/v3/`, `ml/uncertainty/production/v1/`, or `ml/decision_support/production/v1/`.
- **Rule 3 (Synthetic Isolation)**: Synthetic data generated in Phase 7 is an experimental asset and is NOT part of the production system.

---

## 6. Before/After Integrity Verification Procedure

Prior to and immediately following Phase 7B execution, the pipeline must execute an automated cryptographic check:

```python
# Before & After Check in Phase 7B Runner
with open("ml/experiments/phase6f/phase6f_freeze_manifest.json") as f:
    freeze_manifest = json.load(f)

for rel_path, expected_hash in freeze_manifest["artifact_hashes"].items():
    actual_hash = hashlib.sha256(Path(rel_path).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"PHASE 6F FREEZE VIOLATION: {rel_path} was modified!")
```

If any hash differs:
- Phase 7B is immediately aborted with status **`FAILED: PHASE 6F FREEZE VIOLATION`**.

---

## 7. Freeze Acceptance Status

```
============================================================
PHASE 6F FREEZE GATE: ACCEPTANCE VERIFICATION
============================================================

Production model hash:             PASS
Feature registry hash:              PASS
6D uncertainty hash:                PASS
6E artifact integrity:              PASS
6F schema integrity:                PASS
6F decision-rule integrity:        PASS
6F provenance integrity:            PASS
6F manifest integrity:              PASS
Freeze manifest generated:          PASS

============================================================
PHASE 6F: FROZEN
PHASE 7B: AUTHORIZED TO START
============================================================
```
