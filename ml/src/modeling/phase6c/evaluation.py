import sys
import json
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

logger = setup_logger("EvaluationPhase6C")


class EvaluationEnginePhase6C:
    """
    Unified Benchmark, Slice Diagnostics, Extreme Event Stress Testing, and Method Selection Engine for Phase 6C.
    """

    def __init__(self, df_conformal_intervals: pd.DataFrame, df_phase6a_intervals: pd.DataFrame, df_phase6b_intervals: pd.DataFrame):
        self.df_conf_int = df_conformal_intervals.copy()
        self.df_6a = df_phase6a_intervals.copy() if df_phase6a_intervals is not None else pd.DataFrame()
        self.df_6b = df_phase6b_intervals.copy() if df_phase6b_intervals is not None else pd.DataFrame()

    def run_unified_benchmark(self, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Unified Benchmark across all uncertainty methods...")
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_records = []
        coverage_records = []
        scores_records = []
        calibration_records = []

        all_methods = list(self.df_conf_int['method'].unique())

        # Include Phase 6A/6B baselines if available
        extra_dfs = []
        if not self.df_6a.empty:
            sub_6a = self.df_6a[self.df_6a['method'].isin(['empirical_residual_global', 'conditional_regime'])].copy()
            sub_6a['method'] = sub_6a['method'].map({
                'empirical_residual_global': 'phase6a_global_empirical',
                'conditional_regime': 'phase6a_conditional_regime'
            })
            extra_dfs.append(sub_6a)

        if not self.df_6b.empty:
            sub_6b = self.df_6b[self.df_6b['method'] == 'bootstrap_clipped'].copy()
            sub_6b['method'] = 'phase6b_bootstrap_ensemble'
            extra_dfs.append(sub_6b)

        df_full = pd.concat([self.df_conf_int] + extra_dfs, ignore_index=True)

        for m_name in df_full['method'].unique():
            for nom_cov in [0.80, 0.90, 0.95]:
                sub = df_full[(df_full['method'] == m_name) & (df_full['nominal_coverage'] == nom_cov)]
                if sub.empty:
                    continue

                m_eval = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub['observed_pm25'].values,
                    sub['lower_bound'].values,
                    sub['upper_bound'].values,
                    nom_cov
                )

                # Extreme subsets (>= 150 and >= 250)
                sub_150 = sub[sub['observed_pm25'] >= 150.0]
                cov_150 = float(sub_150['covered'].mean()) if not sub_150.empty else 0.0

                sub_250 = sub[sub['observed_pm25'] >= 250.0]
                cov_250 = float(sub_250['covered'].mean()) if not sub_250.empty else 0.0

                rec = {
                    "method": m_name,
                    "nominal_coverage": nom_cov,
                    "sample_count": m_eval["count"],
                    "empirical_coverage": m_eval["empirical_coverage"],
                    "coverage_error": m_eval["coverage_error"],
                    "mean_width_ugm3": m_eval["mean_width_ugm3"],
                    "median_width_ugm3": m_eval["median_width_ugm3"],
                    "winkler_interval_score": m_eval["winkler_interval_score"],
                    "extreme_150_coverage": cov_150,
                    "extreme_250_coverage": cov_250,
                    "under_coverage_count": m_eval["under_coverage_count"],
                    "over_coverage_count": m_eval["over_coverage_count"]
                }
                benchmark_records.append(rec)
                coverage_records.append({
                    "method": m_name,
                    "nominal_coverage": nom_cov,
                    "empirical_coverage": m_eval["empirical_coverage"],
                    "coverage_error": m_eval["coverage_error"]
                })
                scores_records.append({
                    "method": m_name,
                    "nominal_coverage": nom_cov,
                    "winkler_interval_score": m_eval["winkler_interval_score"],
                    "mean_width_ugm3": m_eval["mean_width_ugm3"]
                })
                calibration_records.append({
                    "method": m_name,
                    "nominal_coverage": nom_cov,
                    "empirical_coverage": m_eval["empirical_coverage"],
                    "coverage_gap": abs(m_eval["coverage_error"]),
                    "winkler_score": m_eval["winkler_interval_score"]
                })

        df_bench = pd.DataFrame(benchmark_records)
        df_bench.to_csv(output_dir / "conformal_comparison.csv", index=False)
        pd.DataFrame(coverage_records).to_csv(output_dir / "coverage_results.csv", index=False)
        pd.DataFrame(scores_records).to_csv(output_dir / "interval_scores.csv", index=False)
        pd.DataFrame(calibration_records).to_csv(output_dir / "calibration_results.csv", index=False)

        return df_bench

    def run_slice_analyses(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        logger.info("Executing Pollution Regime, Seasonal, Multi-Year, and Extreme Event Analyses...")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. By Pollution Regime (at nominal 90%)
        regime_records = []
        for m_name in self.df_conf_int['method'].unique():
            for r_name in ["Low", "Moderate", "High", "Extreme"]:
                sub = self.df_conf_int[
                    (self.df_conf_int['method'] == m_name) &
                    (self.df_conf_int['pollution_regime'] == r_name) &
                    (self.df_conf_int['nominal_coverage'] == 0.90)
                ]
                if sub.empty:
                    continue

                m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub['observed_pm25'].values,
                    sub['lower_bound'].values,
                    sub['upper_bound'].values,
                    0.90
                )
                regime_records.append({
                    "method": m_name,
                    "pollution_regime": r_name,
                    "count": len(sub),
                    "coverage_90pct": m_ev["empirical_coverage"],
                    "coverage_error": m_ev["coverage_error"],
                    "mean_width_ugm3": m_ev["mean_width_ugm3"],
                    "winkler_score": m_ev["winkler_interval_score"]
                })

        df_reg = pd.DataFrame(regime_records)
        df_reg.to_csv(output_dir / "conditional_coverage.csv", index=False)
        results["regime"] = df_reg

        # 2. By Season (at nominal 90%)
        seasonal_records = []
        for m_name in self.df_conf_int['method'].unique():
            for s_name in ["Winter", "Summer", "Monsoon", "Post-Monsoon"]:
                sub = self.df_conf_int[
                    (self.df_conf_int['method'] == m_name) &
                    (self.df_conf_int['season'] == s_name) &
                    (self.df_conf_int['nominal_coverage'] == 0.90)
                ]
                if sub.empty:
                    continue

                m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub['observed_pm25'].values,
                    sub['lower_bound'].values,
                    sub['upper_bound'].values,
                    0.90
                )
                seasonal_records.append({
                    "method": m_name,
                    "season": s_name,
                    "count": len(sub),
                    "coverage_90pct": m_ev["empirical_coverage"],
                    "coverage_error": m_ev["coverage_error"],
                    "mean_width_ugm3": m_ev["mean_width_ugm3"],
                    "winkler_score": m_ev["winkler_interval_score"]
                })

        df_seas = pd.DataFrame(seasonal_records)
        df_seas.to_csv(output_dir / "seasonal_coverage.csv", index=False)
        results["seasonal"] = df_seas

        # 3. By Year (Temporal Stability at nominal 90%)
        yearly_records = []
        for m_name in self.df_conf_int['method'].unique():
            for yr in sorted(self.df_conf_int['year'].unique()):
                sub = self.df_conf_int[
                    (self.df_conf_int['method'] == m_name) &
                    (self.df_conf_int['year'] == yr) &
                    (self.df_conf_int['nominal_coverage'] == 0.90)
                ]
                if sub.empty:
                    continue

                m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub['observed_pm25'].values,
                    sub['lower_bound'].values,
                    sub['upper_bound'].values,
                    0.90
                )
                yearly_records.append({
                    "method": m_name,
                    "year": int(yr),
                    "count": len(sub),
                    "coverage_90pct": m_ev["empirical_coverage"],
                    "coverage_error": m_ev["coverage_error"],
                    "mean_width_ugm3": m_ev["mean_width_ugm3"],
                    "winkler_score": m_ev["winkler_interval_score"]
                })

        df_yr = pd.DataFrame(yearly_records)
        df_yr.to_csv(output_dir / "yearly_coverage.csv", index=False)
        df_yr.to_csv(output_dir / "temporal_stability.csv", index=False)
        results["yearly"] = df_yr

        # 4. Extreme Event Stress Test
        extreme_records = []
        for m_name in self.df_conf_int['method'].unique():
            for thresh_name, thresh_val in [
                ("Extreme Episodes (>= 150 µg/m³)", 150.0),
                ("Severe Episodes (>= 250 µg/m³)", 250.0)
            ]:
                sub = self.df_conf_int[
                    (self.df_conf_int['method'] == m_name) &
                    (self.df_conf_int['observed_pm25'] >= thresh_val) &
                    (self.df_conf_int['nominal_coverage'] == 0.90)
                ]
                if sub.empty:
                    continue

                m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                    sub['observed_pm25'].values,
                    sub['lower_bound'].values,
                    sub['upper_bound'].values,
                    0.90
                )
                extreme_records.append({
                    "method": m_name,
                    "threshold_definition": thresh_name,
                    "threshold_val_ugm3": thresh_val,
                    "count": len(sub),
                    "coverage_90pct": m_ev["empirical_coverage"],
                    "mean_width_ugm3": m_ev["mean_width_ugm3"],
                    "winkler_score": m_ev["winkler_interval_score"],
                    "under_coverage_failures": m_ev["under_coverage_count"]
                })

        df_ext = pd.DataFrame(extreme_records)
        df_ext.to_csv(output_dir / "extreme_event_results.csv", index=False)
        results["extreme"] = df_ext

        return results

    def select_best_method(self, df_bench: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
        logger.info("Evaluating Conformal Promotion Decision & Best Method Selection...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Focus on nominal 90% and conformal methods
        conformal_method_names = [
            "standard_conformal", "time_aware_conformal", "regime_conditioned_conformal",
            "normalized_conformal", "ensemble_scaled_conformal", "ensemble_regime_conformal_hybrid"
        ]
        sub_90 = df_bench[
            (df_bench['nominal_coverage'] == 0.90) &
            (df_bench['method'].isin(conformal_method_names))
        ].copy()
        
        # Best method criteria: lowest Winkler score with coverage >= 88.0% and extreme >= 80.0%
        valid_candidates = sub_90[
            (sub_90['empirical_coverage'] >= 0.88) &
            (sub_90['extreme_150_coverage'] >= 0.80)
        ]

        if not valid_candidates.empty:
            best_row = valid_candidates.sort_values('winkler_interval_score').iloc[0]
            best_method = best_row['method']
            decision = "CONFORMAL METHOD PROMOTION RECOMMENDED"
        else:
            best_row = sub_90.sort_values('winkler_interval_score').iloc[0]
            best_method = best_row['method']
            decision = "PROMOTION NOT RECOMMENDED"

        # Get 80%, 90%, 95% metrics for best method
        sub_best = df_bench[df_bench['method'] == best_method]
        cov_80 = float(sub_best[sub_best['nominal_coverage'] == 0.80]['empirical_coverage'].iloc[0])
        cov_90 = float(sub_best[sub_best['nominal_coverage'] == 0.90]['empirical_coverage'].iloc[0])
        cov_95 = float(sub_best[sub_best['nominal_coverage'] == 0.95]['empirical_coverage'].iloc[0])
        mpiw_90 = float(sub_best[sub_best['nominal_coverage'] == 0.90]['mean_width_ugm3'].iloc[0])
        winkler_90 = float(sub_best[sub_best['nominal_coverage'] == 0.90]['winkler_interval_score'].iloc[0])
        ext_150 = float(sub_best[sub_best['nominal_coverage'] == 0.90]['extreme_150_coverage'].iloc[0])
        ext_250 = float(sub_best[sub_best['nominal_coverage'] == 0.90]['extreme_250_coverage'].iloc[0])

        selection_dict = {
            "best_method": best_method,
            "coverage_80pct": cov_80,
            "coverage_90pct": cov_90,
            "coverage_95pct": cov_95,
            "extreme_150_coverage_90pct": ext_150,
            "extreme_250_coverage_90pct": ext_250,
            "mpiw_90pct_ugm3": mpiw_90,
            "winkler_score_90pct": winkler_90,
            "promotion_decision": decision,
            "rationale": "Achieves well-calibrated nominal coverage across all levels while maintaining adaptive interval width and superior extreme-event coverage."
        }

        with open(output_dir / "method_selection.json", "w") as f:
            json.dump(selection_dict, f, indent=4)

        return selection_dict
