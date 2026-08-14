import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.data_v3.external_ingestion import ExternalIngestionV3
from ml.src.data_v3.temporal_alignment import TemporalAlignmentV3
from ml.src.data_v3.spatial_alignment import SpatialAlignmentV3
from ml.src.data_v3.quality_audit import QualityAuditV3
from ml.src.data_v3.leakage_audit import LeakageAuditV3
from ml.src.data_v3.feature_registry import FeatureRegistryV3
from ml.src.data_v3.dataset_builder import DatasetBuilderV3

logger = setup_logger("RunnerDataV3")


def run_data_v3_pipeline():
    logger.info("=== Starting Dataset v3 Data Construction & Audit Pipeline ===")

    output_exp_dir = ROOT_DIR / "ml" / "experiments" / "phase4g"
    output_exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Raw & Process External Data
    ingestion = ExternalIngestionV3()
    proc_df = ingestion.run()

    # 2. Temporal & Spatial Alignment
    temporal = TemporalAlignmentV3()
    spatial = SpatialAlignmentV3()

    v2_df = pd.read_csv(ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv")
    temporal.align_and_validate(v2_df, proc_df)
    spatial.validate_spatial_metadata()

    # 3. Automated Quality Audit
    qa = QualityAuditV3()
    qa.run_audit(proc_df, output_exp_dir)

    # 4. Leakage Audit
    leakage = LeakageAuditV3()
    leakage.run_leakage_audit(proc_df, output_exp_dir)

    # 5. Feature Registry & Data Dictionary
    ext_cols = [c for c in proc_df.columns if c != 'date']
    reg = FeatureRegistryV3()
    reg.build_registry(ext_cols, output_exp_dir)

    # 6. Build Dataset v3 Modeling Dataset
    builder = DatasetBuilderV3()
    v3_df = builder.build_dataset_v3(proc_df)

    logger.info("=== Dataset v3 Data Construction Pipeline Completed Successfully ===")
    return v3_df


if __name__ == "__main__":
    run_data_v3_pipeline()
