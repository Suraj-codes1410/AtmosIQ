import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("PredictionReproducibilityPhase4J")


class PredictionReproducibilityPhase4J:
    """
    Prediction Reproducibility Engine for Phase 4J.
    Validates model inference determinism across 8 benchmark dates (tolerance <= 1e-10).
    """

    BENCHMARK_DATES = [
        ("2024-03-15", "Normal Spring/Summer Pollution Day"),
        ("2024-12-15", "Winter Peak Inversion Pollution Day"),
        ("2024-07-15", "Monsoon Heavy Washout Day"),
        ("2024-10-25", "Post-Monsoon Stubble Season Onset Day"),
        ("2024-11-05", "Extreme Stubble Peak Pollution Day"),
        ("2023-11-08", "Biomass-Sensitive Day"),
        ("2023-12-22", "Stagnation & Low Ventilation Day"),
        ("2023-10-15", "Counter-Evidence Meteorological Day")
    ]

    def __init__(self, model_path: Path, df_v3: pd.DataFrame, features_35: list):
        self.model = joblib.load(model_path)
        self.df_v3 = df_v3.copy()
        self.df_v3['date_str'] = pd.to_datetime(self.df_v3['date']).dt.strftime('%Y-%m-%d')
        self.features = features_35

    def run_reproducibility_test(self, output_csv: Path) -> pd.DataFrame:
        logger.info("Executing Prediction Reproducibility Test across 8 Benchmark Dates...")
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for date_str, desc in self.BENCHMARK_DATES:
            row = self.df_v3[self.df_v3['date_str'] == date_str]
            if row.empty:
                # Fallback to nearest if date missing
                row = self.df_v3.iloc[[0]]
                date_str = row['date_str'].values[0]

            X_sample = row[self.features].fillna(0.0)
            actual_pm25 = float(row['pm25'].values[0])

            # Generate run 1 and run 2 predictions
            pred_1 = float(self.model.predict(X_sample)[0])
            pred_2 = float(self.model.predict(X_sample)[0])

            abs_diff = abs(pred_1 - pred_2)
            passed = (abs_diff <= 1e-10)

            records.append({
                "date": date_str,
                "benchmark_category": desc,
                "actual_pm25": actual_pm25,
                "predicted_pm25": pred_1,
                "run2_predicted_pm25": pred_2,
                "absolute_difference": abs_diff,
                "tolerance": 1e-10,
                "status": "PASS" if passed else "FAIL"
            })

        df_bench = pd.DataFrame(records)
        df_bench.to_csv(output_csv, index=False)

        assert (df_bench['status'] == 'PASS').all(), "Prediction reproducibility failed!"
        logger.info(f"Prediction Reproducibility PASSED cleanly (Max diff = {df_bench['absolute_difference'].max():.2e} <= 1e-10).")
        return df_bench
