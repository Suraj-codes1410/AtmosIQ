# AtmosIQ Phase 10E: Known Model Limitations & Operational Boundary Document

## 1. Categorization of Limitations
This document formally distinguishes intrinsic model limitations from operational faults:

### A. MODEL LIMITATIONS (Inherent Empirical Behavior)
1. **Winter Season Under-Prediction**:
   - During severe surface temperature inversions and boundary layer collapse ($< 300\text{ m}$), the model exhibits an empirical negative bias ($-8.12\,\mu\text{g/m}^3$) and elevated MAE ($42.15\,\mu\text{g/m}^3$).
2. **Emergency Pollution Episodes ($> 250\,\mu\text{g/m}^3$)**:
   - Peak episodic spikes (e.g. agricultural burning and stagnation) exhibit higher residual dispersion (MAE $54.15\,\mu\text{g/m}^3$).
   - Conformal 90% prediction intervals ($\pm 95.66\,\mu\text{g/m}^3$) encompass these variations but widen correspondingly.

### B. DEPLOYMENT & OPERATIONAL SAFEGUARDS
- **Contract Violations & Schema Rejection**: Malformed payloads or missing features are safely rejected with HTTP 400 without producing silent corrupted forecasts.
- **Automated Rollback Policy**: Anomaly or drift severity breach (ORANGE/RED) initiates deterministic rollback to `MODEL_V3_PRODUCTION`.

### C. SCIENTIFIC & PHYSICAL SAFEGUARDS
- **Empirical Uncertainty**: Conformal prediction intervals represent statistical characterization of historical residuals, NOT guaranteed deterministic physical bounds.
- **Statistical Fidelity != Causal Truth**: Deep learning feature mappings do not constitute physical causal proofs of atmospheric transport mechanisms.
