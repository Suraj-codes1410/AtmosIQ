import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6a.coverage_metrics import IntervalEvaluationMetricsPhase6A

logger = setup_logger("RegimeAnalysisPhase6A")


class RegimeAnalysisEnginePhase6A:
    """
    Conditional Coverage Diagnostics & Extreme Episode Uncertainty Engine for Phase 6A.
    Evaluates whether baseline intervals under-cover extreme pollution or show seasonal heterogeneity.
    """

    def __init__(self, df_intervals: pd.DataFrame):
        self.df_intervals = df_intervals.copy()

    def run_conditional_coverage_analysis(self, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Conditional Coverage Diagnostics by Season, Year, Regime, and Extreme Episodes...")
        output_dir.mkdir(parents=True, exist_ok=True)

        records = []
        methods = self.df_intervals['method'].unique()
        nominal_covs = self.df_intervals['nominal_coverage'].unique()

        for method in methods:
            for nom_cov in nominal_covs:
                sub_method = self.df_intervals[
                    (self.df_intervals['method'] == method) &
                    (self.df_intervals['nominal_coverage'] == nom_cov)
                ]

                # 1. Overall
                y_true = sub_method['observed_pm25'].values
                l_arr = sub_method['lower_bound'].values
                u_arr = sub_method['upper_bound'].values
                m_all = IntervalEvaluationMetricsPhase6A.evaluate_interval(y_true, l_arr, u_arr, nom_cov)
                records.append({
                    "method": method,
                    "nominal_coverage": nom_cov,
                    "dimension": "Global",
                    "slice_name": "Overall_All_Folds",
                    "sample_count": len(y_true),
                    "empirical_coverage": m_all["empirical_coverage"],
                    "coverage_error": m_all["coverage_error"],
                    "mean_width_ugm3": m_all["mean_width_ugm3"],
                    "median_width_ugm3": m_all["median_width_ugm3"],
                    "winkler_interval_score": m_all["winkler_interval_score"]
                })

                # 2. By Evaluation Year
                for yr in sorted(sub_method['eval_year'].unique()):
                    sub_yr = sub_method[sub_method['eval_year'] == yr]
                    m_yr = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                        sub_yr['observed_pm25'].values,
                        sub_yr['lower_bound'].values,
                        sub_yr['upper_bound'].values,
                        nom_cov
                    )
                    records.append({
                        "method": method,
                        "nominal_coverage": nom_cov,
                        "dimension": "Year",
                        "slice_name": str(yr),
                        "sample_count": len(sub_yr),
                        "empirical_coverage": m_yr["empirical_coverage"],
                        "coverage_error": m_yr["coverage_error"],
                        "mean_width_ugm3": m_yr["mean_width_ugm3"],
                        "median_width_ugm3": m_yr["median_width_ugm3"],
                        "winkler_interval_score": m_yr["winkler_interval_score"]
                    })

                # 3. By Season
                for s in ["Winter", "Summer", "Monsoon", "Post-Monsoon"]:
                    sub_s = sub_method[sub_method['season'] == s]
                    if not sub_s.empty:
                        m_s = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                            sub_s['observed_pm25'].values,
                            sub_s['lower_bound'].values,
                            sub_s['upper_bound'].values,
                            nom_cov
                        )
                        records.append({
                            "method": method,
                            "nominal_coverage": nom_cov,
                            "dimension": "Season",
                            "slice_name": s,
                            "sample_count": len(sub_s),
                            "empirical_coverage": m_s["empirical_coverage"],
                            "coverage_error": m_s["coverage_error"],
                            "mean_width_ugm3": m_s["mean_width_ugm3"],
                            "median_width_ugm3": m_s["median_width_ugm3"],
                            "winkler_interval_score": m_s["winkler_interval_score"]
                        })

                # 4. By Pollution Regime
                for r in ["Low", "Moderate", "High", "Extreme"]:
                    sub_r = sub_method[sub_method['pollution_regime'] == r]
                    if not sub_r.empty:
                        m_r = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                            sub_r['observed_pm25'].values,
                            sub_r['lower_bound'].values,
                            sub_r['upper_bound'].values,
                            nom_cov
                        )
                        records.append({
                            "method": method,
                            "nominal_coverage": nom_cov,
                            "dimension": "Pollution_Regime",
                            "slice_name": r,
                            "sample_count": len(sub_r),
                            "empirical_coverage": m_r["empirical_coverage"],
                            "coverage_error": m_r["coverage_error"],
                            "mean_width_ugm3": m_r["mean_width_ugm3"],
                            "median_width_ugm3": m_r["median_width_ugm3"],
                            "winkler_interval_score": m_r["winkler_interval_score"]
                        })

                # 5. Extreme Episodes (observed PM2.5 >= 150 µg/m³)
                sub_ext = sub_method[sub_method['observed_pm25'] >= 150.0]
                if not sub_ext.empty:
                    m_ext = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                        sub_ext['observed_pm25'].values,
                        sub_ext['lower_bound'].values,
                        sub_ext['upper_bound'].values,
                        nom_cov
                    )
                    records.append({
                        "method": method,
                        "nominal_coverage": nom_cov,
                        "dimension": "Extreme_Subset",
                        "slice_name": "Extreme_Episodes_ge_150",
                        "sample_count": len(sub_ext),
                        "empirical_coverage": m_ext["empirical_coverage"],
                        "coverage_error": m_ext["coverage_error"],
                        "mean_width_ugm3": m_ext["mean_width_ugm3"],
                        "median_width_ugm3": m_ext["median_width_ugm3"],
                        "winkler_interval_score": m_ext["winkler_interval_score"]
                    })

        df_cond = pd.DataFrame(records)
        df_cond.to_csv(output_dir / "conditional_coverage.csv", index=False)

        logger.info(f"Conditional coverage analysis complete: {len(df_cond)} slice evaluations.")
        return df_cond
