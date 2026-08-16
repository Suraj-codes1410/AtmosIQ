import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6a.coverage_metrics import IntervalEvaluationMetricsPhase6A

logger = setup_logger("RegimeSeasonalAnalysisPhase6B")


class RegimeSeasonalAnalysisEnginePhase6B:
    """
    Pollution Regime, Seasonal, Multi-Year, and Extreme Event Uncertainty Engine for Phase 6B.
    Evaluates ensemble spread expansion and prediction interval coverage across environmental slices.
    """

    def __init__(self, df_summary: pd.DataFrame, df_intervals: pd.DataFrame):
        self.df_summary = df_summary.copy()
        self.df_intervals = df_intervals.copy()

    def run_all_slice_analyses(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        logger.info("Executing Environmental Regime, Seasonal, Multi-Year, and Extreme Event Uncertainty Analyses...")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Pollution Regime Uncertainty
        regime_records = []
        for r_name in ["Low", "Moderate", "High", "Extreme"]:
            sub_sum = self.df_summary[self.df_summary['pollution_regime'] == r_name]
            if sub_sum.empty:
                continue

            n = len(sub_sum)
            mae_val = float(sub_sum['absolute_error'].mean())
            rmse_val = float(np.sqrt((sub_sum['residual'] ** 2).mean()))
            spread_val = float(sub_sum['ensemble_std'].mean())

            # Evaluate 80%, 90%, 95% coverage for clipped intervals
            sub_int = self.df_intervals[
                (self.df_intervals['pollution_regime'] == r_name) &
                (self.df_intervals['method'] == 'bootstrap_clipped')
            ]

            cov_80 = float(sub_int[sub_int['nominal_coverage'] == 0.80]['covered'].mean()) if not sub_int.empty else 0.0
            cov_90 = float(sub_int[sub_int['nominal_coverage'] == 0.90]['covered'].mean()) if not sub_int.empty else 0.0
            cov_95 = float(sub_int[sub_int['nominal_coverage'] == 0.95]['covered'].mean()) if not sub_int.empty else 0.0
            mpiw_90 = float(sub_int[sub_int['nominal_coverage'] == 0.90]['interval_width'].mean()) if not sub_int.empty else 0.0

            # Calculate Winkler Score on 90%
            sub_int_90 = sub_int[sub_int['nominal_coverage'] == 0.90]
            if not sub_int_90.empty:
                m_eval = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub_int_90['observed_pm25'].values,
                    sub_int_90['lower_bound'].values,
                    sub_int_90['upper_bound'].values,
                    0.90
                )
                winkler_score = m_eval['winkler_interval_score']
            else:
                winkler_score = 0.0

            regime_records.append({
                "pollution_regime": r_name,
                "count": n,
                "mae_ugm3": mae_val,
                "rmse_ugm3": rmse_val,
                "mean_ensemble_spread_ugm3": spread_val,
                "coverage_80pct": cov_80,
                "coverage_90pct": cov_90,
                "coverage_95pct": cov_95,
                "mean_width_90pct_ugm3": mpiw_90,
                "winkler_score_90pct": winkler_score
            })

        df_regime = pd.DataFrame(regime_records)
        df_regime.to_csv(output_dir / "regime_uncertainty.csv", index=False)
        results["regime"] = df_regime

        # 2. Seasonal Uncertainty
        seasonal_records = []
        for s_name in ["Winter", "Summer", "Monsoon", "Post-Monsoon"]:
            sub_sum = self.df_summary[self.df_summary['season'] == s_name]
            if sub_sum.empty:
                continue

            n = len(sub_sum)
            mae_val = float(sub_sum['absolute_error'].mean())
            rmse_val = float(np.sqrt((sub_sum['residual'] ** 2).mean()))
            mean_spread = float(sub_sum['ensemble_std'].mean())
            med_spread = float(sub_sum['ensemble_std'].median())

            sub_int = self.df_intervals[
                (self.df_intervals['season'] == s_name) &
                (self.df_intervals['method'] == 'bootstrap_clipped') &
                (self.df_intervals['nominal_coverage'] == 0.90)
            ]

            cov_90 = float(sub_int['covered'].mean()) if not sub_int.empty else 0.0
            mpiw_90 = float(sub_int['interval_width'].mean()) if not sub_int.empty else 0.0

            if not sub_int.empty:
                m_eval = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub_int['observed_pm25'].values,
                    sub_int['lower_bound'].values,
                    sub_int['upper_bound'].values,
                    0.90
                )
                winkler_score = m_eval['winkler_interval_score']
            else:
                winkler_score = 0.0

            seasonal_records.append({
                "season": s_name,
                "count": n,
                "mae_ugm3": mae_val,
                "rmse_ugm3": rmse_val,
                "mean_spread_ugm3": mean_spread,
                "median_spread_ugm3": med_spread,
                "coverage_90pct": cov_90,
                "mean_width_90pct_ugm3": mpiw_90,
                "winkler_score_90pct": winkler_score
            })

        df_seasonal = pd.DataFrame(seasonal_records)
        df_seasonal.to_csv(output_dir / "seasonal_uncertainty.csv", index=False)
        results["seasonal"] = df_seasonal

        # 3. Yearly Uncertainty
        yearly_records = []
        for yr in sorted(self.df_summary['year'].unique()):
            sub_sum = self.df_summary[self.df_summary['year'] == yr]
            n = len(sub_sum)
            mae_val = float(sub_sum['absolute_error'].mean())
            rmse_val = float(np.sqrt((sub_sum['residual'] ** 2).mean()))
            ss_res = float(np.sum(sub_sum['residual'] ** 2))
            ss_tot = float(np.sum((sub_sum['observed_pm25'] - sub_sum['observed_pm25'].mean()) ** 2))
            r2_val = float(1.0 - (ss_res / ss_tot))

            spearman_corr, _ = stats.spearmanr(sub_sum['ensemble_std'], sub_sum['absolute_error'])

            sub_int = self.df_intervals[
                (self.df_intervals['year'] == yr) &
                (self.df_intervals['method'] == 'bootstrap_clipped') &
                (self.df_intervals['nominal_coverage'] == 0.90)
            ]

            cov_90 = float(sub_int['covered'].mean()) if not sub_int.empty else 0.0
            mpiw_90 = float(sub_int['interval_width'].mean()) if not sub_int.empty else 0.0

            if not sub_int.empty:
                m_eval = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub_int['observed_pm25'].values,
                    sub_int['lower_bound'].values,
                    sub_int['upper_bound'].values,
                    0.90
                )
                winkler_score = m_eval['winkler_interval_score']
            else:
                winkler_score = 0.0

            yearly_records.append({
                "year": int(yr),
                "count": n,
                "ensemble_mean_mae_ugm3": mae_val,
                "ensemble_mean_rmse_ugm3": rmse_val,
                "ensemble_mean_r2": r2_val,
                "spearman_spread_error_corr": float(spearman_corr),
                "coverage_90pct": cov_90,
                "mean_width_90pct_ugm3": mpiw_90,
                "winkler_score_90pct": winkler_score
            })

        df_yearly = pd.DataFrame(yearly_records)
        df_yearly.to_csv(output_dir / "yearly_uncertainty.csv", index=False)
        results["yearly"] = df_yearly

        # 4. Extreme Event Uncertainty
        extreme_records = []
        for thresh_name, thresh_val in [
            ("Extreme Episodes (PM2.5 >= 150 µg/m³)", 150.0),
            ("Severe Episodes (PM2.5 >= 250 µg/m³)", 250.0)
        ]:
            sub_sum = self.df_summary[self.df_summary['observed_pm25'] >= thresh_val]
            n = len(sub_sum)
            mae_val = float(sub_sum['absolute_error'].mean()) if n > 0 else 0.0
            rmse_val = float(np.sqrt((sub_sum['residual'] ** 2).mean())) if n > 0 else 0.0
            spread_val = float(sub_sum['ensemble_std'].mean()) if n > 0 else 0.0

            sub_int = self.df_intervals[
                (self.df_intervals['observed_pm25'] >= thresh_val) &
                (self.df_intervals['method'] == 'bootstrap_clipped')
            ]

            cov_80 = float(sub_int[sub_int['nominal_coverage'] == 0.80]['covered'].mean()) if not sub_int.empty else 0.0
            cov_90 = float(sub_int[sub_int['nominal_coverage'] == 0.90]['covered'].mean()) if not sub_int.empty else 0.0
            cov_95 = float(sub_int[sub_int['nominal_coverage'] == 0.95]['covered'].mean()) if not sub_int.empty else 0.0
            mpiw_90 = float(sub_int[sub_int['nominal_coverage'] == 0.90]['interval_width'].mean()) if not sub_int.empty else 0.0

            sub_int_90 = sub_int[sub_int['nominal_coverage'] == 0.90]
            if not sub_int_90.empty:
                m_eval = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub_int_90['observed_pm25'].values,
                    sub_int_90['lower_bound'].values,
                    sub_int_90['upper_bound'].values,
                    0.90
                )
                winkler_score = m_eval['winkler_interval_score']
            else:
                winkler_score = 0.0

            extreme_records.append({
                "threshold_category": thresh_name,
                "threshold_value_ugm3": thresh_val,
                "count": n,
                "mae_ugm3": mae_val,
                "rmse_ugm3": rmse_val,
                "mean_spread_ugm3": spread_val,
                "coverage_80pct": cov_80,
                "coverage_90pct": cov_90,
                "coverage_95pct": cov_95,
                "mean_width_90pct_ugm3": mpiw_90,
                "winkler_score_90pct": winkler_score
            })

        df_extreme = pd.DataFrame(extreme_records)
        df_extreme.to_csv(output_dir / "extreme_event_uncertainty.csv", index=False)
        results["extreme"] = df_extreme

        # 5. Overall Ensemble Metrics & Calibration
        metrics_records = []
        calibration_records = []

        for m_name in self.df_intervals['method'].unique():
            for nom_cov in [0.80, 0.90, 0.95]:
                sub = self.df_intervals[
                    (self.df_intervals['method'] == m_name) &
                    (self.df_intervals['nominal_coverage'] == nom_cov)
                ]
                if sub.empty:
                    continue

                m_eval = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub['observed_pm25'].values,
                    sub['lower_bound'].values,
                    sub['upper_bound'].values,
                    nom_cov
                )

                metrics_records.append({
                    "method": m_name,
                    "nominal_coverage": nom_cov,
                    "sample_count": m_eval["count"],
                    "empirical_coverage": m_eval["empirical_coverage"],
                    "coverage_error": m_eval["coverage_error"],
                    "mean_width_ugm3": m_eval["mean_width_ugm3"],
                    "median_width_ugm3": m_eval["median_width_ugm3"],
                    "winkler_interval_score": m_eval["winkler_interval_score"]
                })

                calibration_records.append({
                    "method": m_name,
                    "nominal_coverage": nom_cov,
                    "empirical_coverage": m_eval["empirical_coverage"],
                    "coverage_error": m_eval["coverage_error"],
                    "mpiw_ugm3": m_eval["mean_width_ugm3"],
                    "winkler_score": m_eval["winkler_interval_score"],
                    "under_coverage_count": m_eval["under_coverage_count"],
                    "over_coverage_count": m_eval["over_coverage_count"]
                })

        df_metrics = pd.DataFrame(metrics_records)
        df_metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False)
        results["metrics"] = df_metrics

        df_calib = pd.DataFrame(calibration_records)
        df_calib.to_csv(output_dir / "ensemble_calibration.csv", index=False)
        results["calibration"] = df_calib

        logger.info("Environmental slice and calibration analyses complete.")
        return results
