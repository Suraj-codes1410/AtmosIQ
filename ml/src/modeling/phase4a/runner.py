import sys
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4a.attribution_grouper import AttributionGrouperPhase4A
from ml.src.modeling.phase4a.package_freezer import PackageFreezerPhase4A
from ml.src.modeling.phase4a.reproducibility import ReproducibilityEnginePhase4A
from ml.src.modeling.phase4a.doc_generator import DocGeneratorPhase4A

logger = setup_logger("MasterRunnerPhase4A")


class MasterRunnerPhase4A:
    """
    AtmosIQ Phase 4A Master Orchestrator.
    Executes Phase 4A Attribution Model Freeze & Reproducibility Package creation.
    """

    def __init__(self):
        self.v1_frozen = Path("ml/data/modeling/v1/feature_dataset_frozen.csv")
        self.v2_frozen = Path("ml/data/modeling/v2/feature_dataset_frozen.csv")

        self.v1_expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
        self.v2_expected_hash = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"

    def calculate_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_dataset_hashes(self):
        """Verifies immutability of Dataset v1 and Dataset v2."""
        logger.info("Verifying Dataset v1 and Dataset v2 SHA-256 hashes...")
        assert self.v1_frozen.exists(), f"Dataset v1 missing: {self.v1_frozen}"
        assert self.v2_frozen.exists(), f"Dataset v2 missing: {self.v2_frozen}"

        v1_hash = self.calculate_sha256(self.v1_frozen)
        v2_hash = self.calculate_sha256(self.v2_frozen)

        if v1_hash != self.v1_expected_hash:
            raise ValueError(f"CRITICAL DISCREPANCY: Dataset v1 modified! Expected {self.v1_expected_hash}, got {v1_hash}")
        if v2_hash != self.v2_expected_hash:
            raise ValueError(f"CRITICAL DISCREPANCY: Dataset v2 modified! Expected {self.v2_expected_hash}, got {v2_hash}")

        logger.info(f"HASHES VERIFIED: Dataset v1 ({v1_hash[:8]}...) & Dataset v2 ({v2_hash[:8]}...). PASS.")

    def run(self):
        """Executes full Phase 4A master pipeline."""
        logger.info("=== Starting AtmosIQ Phase 4A Master Pipeline ===")

        # 1. Verify dataset hashes
        self.verify_dataset_hashes()

        # 2. Build feature registries & attribution groups
        grouper = AttributionGrouperPhase4A()
        grouper.build_registries()

        # 3. Serialize model & create package manifests + checksums
        freezer = PackageFreezerPhase4A()
        freezer.serialize_and_build_manifests()

        # 4. Verify numerical prediction reproducibility
        repro_engine = ReproducibilityEnginePhase4A()
        verification_results = repro_engine.verify_reproducibility()

        # 5. Generate documentation report docs/phase4/phase4a_model_freeze.md
        doc_gen = DocGeneratorPhase4A()
        doc_gen.generate_documentation(verification_results)

        # 6. Post-execution dataset hash verification
        self.verify_dataset_hashes()

        logger.info("=== Phase 4A Master Pipeline Completed Successfully ===")


if __name__ == "__main__":
    runner = MasterRunnerPhase4A()
    runner.run()
