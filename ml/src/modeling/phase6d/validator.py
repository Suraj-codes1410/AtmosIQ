import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6d.config import ValidationConfigPhase6D
from ml.src.modeling.phase6a.coverage_metrics import IntervalEvaluationMetricsPhase6A
from ml.src.modeling.phase6c.conformal_engine import ConformalPredictionEnginePhase6C

logger = setup_logger("ValidatorPhase6D")


class RevalidationEnginePhase6D:
    """
    Independent Revalidation Engine for Phase 6D.
    Re-runs Normalized Heteroscedastic Conformal Prediction across 2022–2024 (N=1,096)
    and verifies that headline metrics match Phase 6C within tight tolerance.
    """

    def __init__(self, df_v3: pd.DataFrame, features_35: List[str], config: ValidationConfigPhase6D):
        self.df_v3 = df_v3.copy()
        self.df_v3['date_dt'] = pd.to_datetime(self.df_v3['date'])
        self.df_v3 = self.df_v3.sort_values('date_dt').reset_index(drop=True)
        self.features = features_35
        self.config = config

    def run_revalidation(
        self,
        df_control: pd.DataFrame,
        df_boot_preds: pd.DataFrame,
        output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        logger.info("Executing Independent Revalidation of normalized_conformal...")
        output_dir.mkdir(parents=True, exist_ok=True)

        from ml.src.modeling.phase6c.config import ConformalConfigPhase6C
        p6c_config = ConformalConfigPhase6C()
        engine = ConformalPredictionEnginePhase6C(p6c_config)

        df_preds, df_intervals = engine.run_all_conformal_methods(
            self.df_v3,
            self.features,
            df_control,
            df_boot_preds
        )

        sub_norm = df_intervals[df_intervals['method'] == 'normalized_conformal'].copy()
        
        reval_records = []
        for nom_cov in [0.80, 0.90, 0.95]:
            sub = sub_norm[sub_norm['nominal_coverage'] == nom_cov]
            m_ev = IntervalEvaluationMetricsPhase6A.evaluate_interval(
                sub['observed_pm25'].values,
                sub['lower_bound'].values,
                sub['upper_bound'].values,
                nom_cov
            )

            sub_150 = sub[sub['observed_pm25'] >= 150.0]
            cov_150 = float(sub_150['covered'].mean()) if not sub_150.empty else 0.0

            sub_250 = sub[sub['observed_pm25'] >= 250.0]
            cov_250 = float(sub_250['covered'].mean()) if not sub_250.empty else 0.0

            reval_records.append({
                "method": "normalized_conformal",
                "nominal_coverage": nom_cov,
                "sample_count": m_ev["count"],
                "empirical_coverage": m_ev["empirical_coverage"],
                "coverage_error": m_ev["coverage_error"],
                "mean_width_ugm3": m_ev["mean_width_ugm3"],
                "median_width_ugm3": m_ev["median_width_ugm3"],
                "winkler_interval_score": m_ev["winkler_interval_score"],
                "extreme_150_coverage": cov_150,
                "extreme_250_coverage": cov_250,
                "under_coverage_count": m_ev["under_coverage_count"],
                "over_coverage_count": m_ev["over_coverage_count"]
            })

        df_reval = pd.DataFrame(reval_records)
        df_reval.to_csv(output_dir / "phase6d_revalidation.csv", index=False)

        # Extract 90% metrics
        row_90 = df_reval[df_reval['nominal_coverage'] == 0.90].iloc[0]
        cov_90 = row_90['empirical_coverage']
        mpiw_90 = row_90['mean_width_ugm3']
        winkler_90 = row_90['winkler_interval_score']
        cov_ext150 = row_90['extreme_150_coverage']
        cov_ext250 = row_90['extreme_250_coverage']

        logger.info(f"Revalidation Metrics (90% Nominal): Coverage={cov_90*100:.2f}%, MPIW={mpiw_90:.2f} µg/m³, Winkler={winkler_90:.2f}, Extreme(>=150)={cov_ext150*100:.2f}%, Severe(>=250)={cov_ext250*100:.2f}%")

        # Tolerances for validation
        assert abs(cov_90 - 0.8978) < 0.02, f"Revalidation 90% coverage mismatch: {cov_90}"
        assert abs(mpiw_90 - 68.77) < 2.0, f"Revalidation 90% MPIW mismatch: {mpiw_90}"
        assert abs(cov_ext150 - 0.8945) < 0.02, f"Revalidation extreme coverage mismatch: {cov_ext150}"
        assert abs(cov_ext250 - 0.8901) < 0.02, f"Revalidation severe coverage mismatch: {cov_ext250}"

        reval_summary = {
            "revalidation_status": "PASS",
            "coverage_80pct": float(df_reval[df_reval['nominal_coverage'] == 0.80]['empirical_coverage'].iloc[0]),
            "coverage_90pct": float(cov_90),
            "coverage_95pct": float(df_reval[df_reval['nominal_coverage'] == 0.95]['empirical_coverage'].iloc[0]),
            "mpiw_90pct_ugm3": float(mpiw_90),
            "winkler_score_90pct": float(winkler_90),
            "extreme_150_coverage_90pct": float(cov_ext150),
            "extreme_250_coverage_90pct": float(cov_ext250)
        }

        return df_reval, sub_norm, reval_summary
