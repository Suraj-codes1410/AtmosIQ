"""
AtmosIQ Phase 8B: Batch Acceptance Gate & Quality Scoring Engine.
"""

from typing import Dict, Any, Tuple
import pandas as pd


class BatchAcceptanceGate:
    """Evaluates individual batch health and assigns formal acceptance status."""

    def __init__(self):
        pass

    def evaluate_batch(
        self,
        batch_meta: Dict[str, Any],
        fidelity_report: Dict[str, Any],
        mem_report: Dict[str, Any],
        ood_report: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        checks = []

        # 1. Physical Validity & Zero Leakage (Hard Blockers)
        exact_dups = mem_report.get("exact_duplicate_count", 0)
        checks.append({
            "dimension": "Memorization & Duplicates",
            "metric": "Exact duplicate count",
            "value": f"{exact_dups}",
            "status": "PASS" if exact_dups == 0 else "FAIL",
            "criticality": "HARD_BLOCKER"
        })

        # 2. Acceptance Rate
        acc_rate = batch_meta.get("acceptance_rate_pct", 0.0)
        checks.append({
            "dimension": "Extreme-Tail Filter Acceptance",
            "metric": "Acceptance rate %",
            "value": f"{acc_rate:.1f}%",
            "status": "PASS" if acc_rate >= 50.0 else "WARNING",
            "criticality": "PRIMARY"
        })

        # 3. Distribution Fidelity (Normalized W1)
        w1_val = fidelity_report.get("mean_normalized_w1", 0.0)
        checks.append({
            "dimension": "Distributional Fidelity",
            "metric": "Mean normalized W1",
            "value": f"{w1_val:.4f}",
            "status": "PASS" if w1_val <= 0.60 else "WARNING",
            "criticality": "PRIMARY"
        })

        # 4. Multivariate Correlation (Frobenius Distance)
        frob_val = fidelity_report.get("frobenius_correlation_distance", 0.0)
        checks.append({
            "dimension": "Multivariate Structure",
            "metric": "Frobenius correlation distance",
            "value": f"{frob_val:.4f}",
            "status": "PASS" if frob_val <= 0.30 else "WARNING",
            "criticality": "PRIMARY"
        })

        # 5. Temporal Dynamics (ACF Error Lags 1-7)
        acf_err = fidelity_report.get("mean_acf_error_lags_1_7", 0.0)
        checks.append({
            "dimension": "Temporal Dynamics",
            "metric": "Mean ACF error (Lags 1-7)",
            "value": f"{acf_err:.4f}",
            "status": "PASS" if acf_err <= 0.25 else "WARNING",
            "criticality": "PRIMARY"
        })

        # 6. OOD Outlier Density
        outlier_pct = ood_report.get("outlier_pct", 0.0)
        checks.append({
            "dimension": "Feature-Space OOD Support",
            "metric": "Outlier percentage",
            "value": f"{outlier_pct:.1f}%",
            "status": "PASS" if outlier_pct <= 50.0 else "WARNING",
            "criticality": "SECONDARY"
        })

        df_checks = pd.DataFrame(checks)

        # Decision rule
        hard_failed = (df_checks[df_checks["criticality"] == "HARD_BLOCKER"]["status"] == "FAIL").any()
        primary_warnings = (df_checks[df_checks["criticality"] == "PRIMARY"]["status"] == "WARNING").sum()

        if hard_failed:
            decision = "REJECT"
        elif primary_warnings > 0:
            decision = "CONDITIONAL_ACCEPT"
        else:
            decision = "ACCEPT"

        report = {
            "batch_id": batch_meta["batch_id"],
            "decision": decision,
            "checks": checks,
        }

        return decision, report
