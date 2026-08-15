import numpy as np
from typing import Dict, Any


class IntervalEvaluationMetricsPhase6A:
    """
    Standard Evaluation Metrics for Prediction Intervals in Phase 6A.
    Computes Empirical Coverage, Coverage Error, Mean Width, Median Width, and Winkler Interval Score.
    """

    @staticmethod
    def evaluate_interval(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        nominal_coverage: float
    ) -> Dict[str, float]:
        n = len(y_true)
        if n == 0:
            return {
                "count": 0,
                "nominal_coverage": nominal_coverage,
                "empirical_coverage": 0.0,
                "coverage_error": 0.0,
                "mean_width_ugm3": 0.0,
                "median_width_ugm3": 0.0,
                "winkler_interval_score": 0.0,
                "under_coverage_count": 0,
                "over_coverage_count": 0
            }

        # 1. Coverage
        covered = (y_true >= lower) & (y_true <= upper)
        emp_coverage = float(np.mean(covered))
        cov_error = float(emp_coverage - nominal_coverage)

        # 2. Widths
        widths = upper - lower
        mean_w = float(np.mean(widths))
        median_w = float(np.median(widths))

        # 3. Winkler Interval Score
        alpha = 1.0 - nominal_coverage
        penalty_factor = 2.0 / alpha

        below_lower = y_true < lower
        above_upper = y_true > upper

        lower_penalties = np.where(below_lower, penalty_factor * (lower - y_true), 0.0)
        upper_penalties = np.where(above_upper, penalty_factor * (y_true - upper), 0.0)

        interval_scores = widths + lower_penalties + upper_penalties
        mean_score = float(np.mean(interval_scores))

        return {
            "count": n,
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": emp_coverage,
            "coverage_error": cov_error,
            "mean_width_ugm3": mean_w,
            "median_width_ugm3": median_w,
            "winkler_interval_score": mean_score,
            "under_coverage_count": int(np.sum(below_lower)),
            "over_coverage_count": int(np.sum(above_upper))
        }
