import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger
from ml.src.features.feature_pipeline import FeatureEngineeringPipeline

logger = setup_logger("FeaturePipelineV2")


class FeaturePipelineEngineV2:
    """
    AtmosIQ Dataset v2 Feature Engineering Pipeline.
    Constructs the 7-level structured feature hierarchy across 1,827 daily observations (2020-2024).
    Enforces strict target leakage prevention (target shifted by >= 1 day for all historical features).
    """

    def __init__(self):
        self.inter_file = Path("ml/data/intermediate/v2/master_dataset_v2.csv")
        self.output_file = Path("ml/data/processed/v2/feature_dataset_v2.csv")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.pipeline = FeatureEngineeringPipeline(input_path=str(self.inter_file), output_path=str(self.output_file))

    def run(self) -> pd.DataFrame:
        """Executes feature engineering pipeline for Dataset v2."""
        logger.info("=== Starting Dataset v2 Feature Engineering Pipeline ===")
        assert self.inter_file.exists(), f"Intermediate master dataset missing: {self.inter_file}"

        df = pd.read_csv(self.inter_file)
        logger.info(f"Loaded intermediate master dataset v2: {len(df)} rows x {len(df.columns)} columns.")

        df_feat = self.pipeline.transform(df)

        assert len(df_feat) == len(df), f"Row count changed during feature extraction: {len(df_feat)} vs {len(df)}"
        assert df_feat.isnull().sum().sum() == 0, "Processed Dataset v2 contains NaNs!"

        df_feat.to_csv(self.output_file, index=False)
        logger.info(f"Processed Feature Dataset v2 saved to: {self.output_file} ({len(df_feat)} rows x {len(df_feat.columns)} columns).")

        return df_feat


if __name__ == "__main__":
    pipeline = FeaturePipelineEngineV2()
    pipeline.run()
