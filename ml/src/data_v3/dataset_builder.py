import sys
import json
import hashlib
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("DatasetBuilderV3")


class DatasetBuilderV3:
    """
    Dataset Builder for Dataset v3.
    Merges immutable Dataset v2 with external environmental features to build Dataset v3.
    Generates dataset_v3_manifest.json and checksums_v3.txt.
    """

    def __init__(self, v3_dir: str = "ml/data/modeling/v3"):
        self.v3_dir = Path(v3_dir)
        self.v3_dir.mkdir(parents=True, exist_ok=True)
        self.v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"

    def build_dataset_v3(self, ext_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building Dataset v3 (Merging Dataset v2 + Processed External Features)...")

        v2_df = pd.read_csv(self.v2_path)
        assert len(v2_df) == 1827, f"Dataset v2 row count mismatch: {len(v2_df)}"

        # Merge on 'date' column
        v3_df = pd.merge(v2_df, ext_df, on="date", how="inner")
        assert len(v3_df) == 1827, f"Merged Dataset v3 row count mismatch: {len(v3_df)}"

        v3_csv_path = self.v3_dir / "feature_dataset_frozen.csv"
        v3_df.to_csv(v3_csv_path, index=False)
        logger.info(f"Dataset v3 frozen dataset saved to {v3_csv_path} ({v3_df.shape[0]} rows, {v3_df.shape[1]} columns).")

        # Also save dataset_v3_prediction_safe.csv
        v3_safe_path = self.v3_dir / "dataset_v3_prediction_safe.csv"
        v3_df.to_csv(v3_safe_path, index=False)

        # Generate SHA-256 Hashes and Checksums
        sha256 = hashlib.sha256()
        with open(v3_csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        v3_hash = sha256.hexdigest()

        checksum_file = self.v3_dir / "checksums_v3.txt"
        with open(checksum_file, "w") as f:
            f.write(f"feature_dataset_frozen.csv  {v3_hash}\n")

        manifest = {
            "dataset_version": "v3.0.0",
            "dataset_name": "AtmosIQ Dataset v3 (5-Year PM2.5 + External Environmental Observations)",
            "date_created": "2026-08-14",
            "sha256_hash": v3_hash,
            "row_count": len(v3_df),
            "col_count": len(v3_df.columns),
            "date_range": {
                "start": v3_df['date'].min(),
                "end": v3_df['date'].max()
            },
            "external_feature_groups": ["precipitation", "pbl_height", "aerosol_optical_depth", "transport_winds"],
            "lineage": "Dataset v1 -> Dataset v2 -> Dataset v3"
        }
        manifest_file = self.v3_dir / "dataset_v3_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Dataset v3 SHA-256 Hash: {v3_hash}")
        logger.info(f"Dataset v3 Manifest saved to {manifest_file}.")

        return v3_df
