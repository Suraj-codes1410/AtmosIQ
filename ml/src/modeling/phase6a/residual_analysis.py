import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ResidualAnalysisPhase6A")


class ResidualAnalysisEnginePhase6A:
    """
    Comprehensive Empirical Residual Distribution Analysis Engine for Phase 6A.
    Computes summary metrics, quantiles, dispersion, higher-order moments, and normality diagnostics.
    """

    def __init__(self, df_preds: pd.DataFrame):
        self.df_preds = df_preds.copy()

    @staticmethod
    def calculate_subset_stats(residuals: np.ndarray, subset_name: str, category: str) -> Dict[str, Any]:
        n = len(residuals)
        if n == 0:
            return {"subset_category": category, "subset_name": subset_name, "count": 0}

        mean_res = float(np.mean(residuals))
        median_res = float(np.median(residuals))
        std_res = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
        mad_res = float(np.median(np.abs(residuals - median_res)))
        mae_res = float(np.mean(np.abs(residuals)))
        medae_res = float(np.median(np.abs(residuals)))
        rmse_res = float(np.sqrt(np.mean(residuals ** 2)))

        min_res = float(np.min(residuals))
        max_res = float(np.max(residuals))

        q01 = float(np.percentile(residuals, 1))
        q05 = float(np.percentile(residuals, 5))
        q10 = float(np.percentile(residuals, 10))
        q25 = float(np.percentile(residuals, 25))
        q50 = float(np.percentile(residuals, 50))
        q75 = float(np.percentile(residuals, 75))
        q90 = float(np.percentile(residuals, 90))
        q95 = float(np.percentile(residuals, 95))
        q99 = float(np.percentile(residuals, 99))

        skew_res = float(stats.skew(residuals)) if n > 2 else 0.0
        kurt_res = float(stats.kurtosis(residuals)) if n > 3 else 0.0

        # Normality Test (D'Agostino's K^2 omnibus test)
        if n >= 20:
            stat_norm, p_norm = stats.normaltest(residuals)
            is_gaussian = bool(p_norm > 0.05)
        else:
            p_norm = 1.0
            is_gaussian = True

        return {
            "subset_category": category,
            "subset_name": subset_name,
            "count": n,
            "mean_residual": mean_res,
            "median_residual": median_res,
            "std_residual": std_res,
            "mad_residual": mad_res,
            "mae_ugm3": mae_res,
            "medae_ugm3": medae_res,
            "rmse_ugm3": rmse_res,
            "min_residual": min_res,
            "max_residual": max_res,
            "q01": q01,
            "q05": q05,
            "q10": q10,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "skewness": skew_res,
            "kurtosis": kurt_res,
            "normality_p_value": float(p_norm),
            "gaussian_assumption_valid": is_gaussian
        }

    def run_comprehensive_analysis(self, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Empirical Residual Distribution Analysis across Regimes, Seasons, and Years...")
        output_dir.mkdir(parents=True, exist_ok=True)

        records = []

        # 1. Global Overall Out-of-Sample Residuals
        res_all = self.df_preds['residual'].values
        records.append(self.calculate_subset_stats(res_all, "Overall_Out_of_Sample", "Global"))

        # 2. By Evaluation Year
        for yr in sorted(self.df_preds['year'].unique()):
            res_yr = self.df_preds[self.df_preds['year'] == yr]['residual'].values
            records.append(self.calculate_subset_stats(res_yr, f"Year_{yr}", "Year"))

        # 3. By Season
        for s in ["Winter", "Summer", "Monsoon", "Post-Monsoon"]:
            sub = self.df_preds[self.df_preds['season'] == s]
            if not sub.empty:
                records.append(self.calculate_subset_stats(sub['residual'].values, s, "Season"))

        # 4. By Pollution Regime
        for r in ["Low", "Moderate", "High", "Extreme"]:
            sub = self.df_preds[self.df_preds['pollution_regime'] == r]
            if not sub.empty:
                records.append(self.calculate_subset_stats(sub['residual'].values, r, "Pollution_Regime"))

        # 5. Extreme Pollution Episode Subset (>= 150 µg/m³)
        sub_ext = self.df_preds[self.df_preds['is_extreme_episode']]
        if not sub_ext.empty:
            records.append(self.calculate_subset_stats(sub_ext['residual'].values, "Extreme_Episodes_ge_150", "Extreme_Episodes"))

        # 6. Severe Pollution Episode Subset (>= 250 µg/m³)
        sub_sev = self.df_preds[self.df_preds['is_severe_episode']]
        if not sub_sev.empty:
            records.append(self.calculate_subset_stats(sub_sev['residual'].values, "Severe_Episodes_ge_250", "Severe_Episodes"))

        df_res_stats = pd.DataFrame(records)
        df_res_stats.to_csv(output_dir / "residual_analysis.csv", index=False)

        logger.info(f"Residual analysis completed: {len(df_res_stats)} demographic subsets evaluated.")
        return df_res_stats
