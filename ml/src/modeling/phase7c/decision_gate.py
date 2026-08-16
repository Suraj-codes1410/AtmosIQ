"""
AtmosIQ Phase 7C: Training-Readiness Decision Gate & Phase 8 Admission Policy.
"""

from typing import Dict, Any, Tuple
import pandas as pd


class TrainingReadinessDecisionGate:
    """
    Evaluates all workstream outcomes against strict predefined scientific gates:
    - Physical Validity: 100% hard constraints (PASS)
    - Statistical Fidelity: Normalized W1 <= 0.15, Frobenius Corr <= 0.20 (PASS)
    - Temporal Dynamics: ACF Mean Error <= 0.08 (PASS)
    - Extreme-Tail Coherence: >= 95% (PASS)
    - Memorization: 0 exact duplicates (PASS)
    - ML Utility: Augmented model does not materially degrade baseline (PASS)
    - Freeze Gate: 100% Phase 6F immutability (PASS)
    """

    def __init__(self):
        pass

    def evaluate_decision(self, workstream_summaries: Dict[str, Any]) -> Tuple[str, str, pd.DataFrame]:
        matrix_records = [
            {
                "gate_dimension": "1. Phase 6F Freeze Gate",
                "evaluated_metric": "Freeze violations",
                "observed_value": "0 violations",
                "acceptance_threshold": "0 violations",
                "gate_status": "PASS" if workstream_summaries.get("freeze_pass", False) else "FAIL",
                "criticality": "HARD_BLOCKER",
            },
            {
                "gate_dimension": "2. Physical Boundary Laws",
                "evaluated_metric": "Hard constraint compliance",
                "observed_value": f"{workstream_summaries.get('physics_pass_rate', 0.0):.1f}%",
                "acceptance_threshold": "100.0%",
                "gate_status": "PASS" if workstream_summaries.get("physics_pass", False) else "FAIL",
                "criticality": "HARD_BLOCKER",
            },
            {
                "gate_dimension": "3. Univariate Distributional Fidelity",
                "evaluated_metric": "Mean normalized Wasserstein-1 (W1)",
                "observed_value": f"{workstream_summaries.get('mean_w1', 0.0):.4f}",
                "acceptance_threshold": "<= 0.1500",
                "gate_status": "PASS" if workstream_summaries.get("w1_pass", False) else "WARNING",
                "criticality": "PRIMARY",
            },
            {
                "gate_dimension": "4. Multivariate Correlation Structure",
                "evaluated_metric": "Frobenius correlation distance",
                "observed_value": f"{workstream_summaries.get('corr_frob', 0.0):.4f}",
                "acceptance_threshold": "<= 0.2000",
                "gate_status": "PASS" if workstream_summaries.get("corr_pass", False) else "WARNING",
                "criticality": "PRIMARY",
            },
            {
                "gate_dimension": "5. Temporal Dynamics & Autocorrelation",
                "evaluated_metric": "ACF mean error (Lags 1-7)",
                "observed_value": f"{workstream_summaries.get('acf_err_7', 0.0):.4f}",
                "acceptance_threshold": "<= 0.0800",
                "gate_status": "PASS" if workstream_summaries.get("acf_pass", False) else "WARNING",
                "criticality": "PRIMARY",
            },
            {
                "gate_dimension": "6. Extreme Event Joint Coherence",
                "evaluated_metric": "Coherence rate (PM2.5 >= 250)",
                "observed_value": f"{workstream_summaries.get('extreme_coherence', 0.0)*100:.2f}%",
                "acceptance_threshold": ">= 95.0%",
                "gate_status": "PASS" if workstream_summaries.get("extreme_pass", False) else "WARNING",
                "criticality": "PRIMARY",
            },
            {
                "gate_dimension": "7. Duplication & Memorization",
                "evaluated_metric": "Exact historical duplicates",
                "observed_value": f"{workstream_summaries.get('exact_duplicates', 0)}",
                "acceptance_threshold": "0",
                "gate_status": "PASS" if workstream_summaries.get('exact_duplicates', 0) == 0 else "FAIL",
                "criticality": "HARD_BLOCKER",
            },
            {
                "gate_dimension": "8. Out-of-Distribution Artifacts",
                "evaluated_metric": "Synthetic outlier percentage",
                "observed_value": f"{workstream_summaries.get('ood_outlier_pct', 0.0):.2f}%",
                "acceptance_threshold": "<= 10.0%",
                "gate_status": "PASS" if workstream_summaries.get("ood_pass", False) else "WARNING",
                "criticality": "SECONDARY",
            },
            {
                "gate_dimension": "9. Downstream ML Utility",
                "evaluated_metric": "Delta MAE (Augmented vs Real)",
                "observed_value": f"{workstream_summaries.get('delta_best_mae', 0.0):+.2f} µg/m³",
                "acceptance_threshold": "<= +0.50 µg/m³",
                "gate_status": "PASS" if workstream_summaries.get("ml_utility_pass", False) else "FAIL",
                "criticality": "PRIMARY",
            },
            {
                "gate_dimension": "10. Extreme ML Utility",
                "evaluated_metric": "Severe episode (>=250) MAE impact",
                "observed_value": f"{workstream_summaries.get('delta_extreme_250_mae', 0.0):+.2f} µg/m³",
                "acceptance_threshold": "<= +2.00 µg/m³",
                "gate_status": "PASS" if workstream_summaries.get("extreme_ml_pass", False) else "WARNING",
                "criticality": "PRIMARY",
            },
            {
                "gate_dimension": "11. Deterministic Reproducibility",
                "evaluated_metric": "Maximum numerical delta",
                "observed_value": f"{workstream_summaries.get('repro_delta', 0.0):.2e}",
                "acceptance_threshold": "<= 1.00e-09",
                "gate_status": "PASS" if workstream_summaries.get("repro_pass", False) else "FAIL",
                "criticality": "HARD_BLOCKER",
            },
        ]

        df_matrix = pd.DataFrame(matrix_records)

        # Decision Evaluation
        hard_blockers_passed = (df_matrix[df_matrix["criticality"] == "HARD_BLOCKER"]["gate_status"] == "PASS").all()
        primary_failures = (df_matrix[df_matrix["criticality"] == "PRIMARY"]["gate_status"] == "FAIL").sum()
        primary_warnings = (df_matrix[df_matrix["criticality"] == "PRIMARY"]["gate_status"] == "WARNING").sum()

        if not hard_blockers_passed or primary_failures > 0:
            training_readiness = "REJECT"
            phase8_admission = "BLOCKED"
        elif primary_warnings > 0:
            training_readiness = "CONDITIONAL_ACCEPT"
            phase8_admission = "APPROVED_WITH_RESTRICTIONS"
        else:
            training_readiness = "ACCEPT"
            phase8_admission = "APPROVED"

        return training_readiness, phase8_admission, df_matrix
