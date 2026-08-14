# AtmosIQ Production Release Notes (v3.0.0)

## Promotion Summary
- **Promoted Candidate**: Random Forest Regressor trained on Dataset v3 with 35 prediction-safe features.
- **Improvement over Phase 3G Control**: ΔMAE = -8.6428 µg/m³ (p = 3.5567e-33, 95% Bootstrap CI: [-9.7750, -7.2943] µg/m³).
- **Extreme Event Improvement**: 17.44 µg/m³ error reduction on PM2.5 >= 150 µg/m³ events.
- **Attribution Revalidation**: TreeSHAP reconstruction error <= 1e-12 µg/m³, 94.73% active-driver counterfactual consistency.
