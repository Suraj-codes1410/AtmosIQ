"""
AtmosIQ Phase 7B: Deterministic Reproducibility Audit Engine.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import logging

from .config import SyntheticConfigPhase7B
from .trajectory_generator import TrajectoryGeneratorPhase7B

logger = logging.getLogger(__name__)


class ReproducibilityAuditorPhase7B:
    """
    Executes duplicate generator runs and validates deterministic equivalence.
    """

    def __init__(self, config: SyntheticConfigPhase7B, feature_registry: List[str]):
        self.config = config
        self.feature_registry = list(feature_registry)

    def run_reproducibility_audit(self, df_train: pd.DataFrame) -> Tuple[bool, float, pd.DataFrame]:
        logger.info("Executing Phase 7B Generator Double-Run Reproducibility Audit...")

        # Run 1
        gen1 = TrajectoryGeneratorPhase7B(self.config, self.feature_registry)
        gen1.fit_from_training_data(df_train)
        df_run1 = gen1.generate_all_trajectories()

        # Run 2
        gen2 = TrajectoryGeneratorPhase7B(self.config, self.feature_registry)
        gen2.fit_from_training_data(df_train)
        df_run2 = gen2.generate_all_trajectories()

        # Compare row counts
        if len(df_run1) != len(df_run2):
            return False, 999.0, pd.DataFrame()

        # Compare numerical columns
        eval_cols = self.feature_registry + ["pm25"]
        max_delta = 0.0
        delta_records = []

        for col in eval_cols:
            v1 = df_run1[col].values
            v2 = df_run2[col].values
            col_max_delta = float(np.max(np.abs(v1 - v2)))
            max_delta = max(max_delta, col_max_delta)
            delta_records.append({
                "feature": col,
                "max_absolute_delta": col_max_delta,
                "status": "PASS" if col_max_delta <= 1e-10 else "FAIL"
            })

        df_delta = pd.DataFrame(delta_records)
        is_reproducible = (max_delta <= 1e-10)
        logger.info(f"Phase 7B Reproducibility Audit completed. Max delta: {max_delta:.2e}, Passed: {is_reproducible}")

        return is_reproducible, max_delta, df_delta
