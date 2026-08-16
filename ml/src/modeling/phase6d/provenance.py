import hashlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ProvenancePhase6D")


class ProvenanceVerifierPhase6D:
    """
    Cryptographic Provenance Verifier for Phase 6D.
    Verifies immutable hashes of Datasets v1, v2, v3, Control Model, and Production v3 Model.
    """

    EXPECTED_HASHES = {
        "v1_dataset": "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df",
        "v2_dataset": "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301",
        "v3_dataset": "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736",
        "v1_control_model": "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162",
        "v3_production_model": "9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210"
    }

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir
        self.v1_dataset_path = self.root_dir / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
        self.v2_dataset_path = self.root_dir / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
        self.v3_dataset_path = self.root_dir / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
        self.v1_control_model_path = self.root_dir / "ml" / "models" / "attribution" / "v1" / "model.joblib"
        self.v3_production_model_path = self.root_dir / "ml" / "models" / "production" / "v3" / "model.joblib"

    @staticmethod
    def calculate_sha256(filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_all(self) -> dict:
        logger.info("Verifying Upstream Immutable Artifact Hashes for Phase 6D...")
        
        v1_h = self.calculate_sha256(self.v1_dataset_path)
        v2_h = self.calculate_sha256(self.v2_dataset_path)
        v3_h = self.calculate_sha256(self.v3_dataset_path)
        v1_m_h = self.calculate_sha256(self.v1_control_model_path)
        v3_m_h = self.calculate_sha256(self.v3_production_model_path)

        checks = {
            "Dataset v1 Integrity": (v1_h == self.EXPECTED_HASHES["v1_dataset"], v1_h, self.EXPECTED_HASHES["v1_dataset"]),
            "Dataset v2 Integrity": (v2_h == self.EXPECTED_HASHES["v2_dataset"], v2_h, self.EXPECTED_HASHES["v2_dataset"]),
            "Dataset v3 Integrity": (v3_h == self.EXPECTED_HASHES["v3_dataset"], v3_h, self.EXPECTED_HASHES["v3_dataset"]),
            "Control Model Integrity": (v1_m_h == self.EXPECTED_HASHES["v1_control_model"], v1_m_h, self.EXPECTED_HASHES["v1_control_model"]),
            "Production v3 Model Integrity": (v3_m_h == self.EXPECTED_HASHES["v3_production_model"], v3_m_h, self.EXPECTED_HASHES["v3_production_model"])
        }

        all_passed = True
        for name, (passed, actual, expected) in checks.items():
            if not passed:
                logger.error(f"{name} FAILED! Actual: {actual}, Expected: {expected}")
                all_passed = False
            else:
                logger.info(f"{name}: PASS ({actual[:16]}...)")

        assert all_passed, "Upstream provenance verification failed in Phase 6D!"
        logger.info("ALL UPSTREAM ARTIFACT HASHES VERIFIED: PASS.")
        return {
            "v1_dataset_hash": v1_h,
            "v2_dataset_hash": v2_h,
            "v3_dataset_hash": v3_h,
            "control_model_hash": v1_m_h,
            "production_model_hash": v3_m_h,
            "all_passed": all_passed
        }
