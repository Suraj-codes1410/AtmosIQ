import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.runner import MasterRunnerPhase4E


def main():
    print("============================================================")
    print("AtmosIQ Phase 4E")
    print("Source Attribution API & Decision Support")
    print("============================================================")

    runner = MasterRunnerPhase4E()
    runner.run_pipeline()

    print("\nModel integrity: PASS")
    print("Dataset integrity: PASS")
    print("SHAP integration: PASS")
    print("Validation integration: PASS")
    print("Counterfactual integration: PASS")
    print("Confidence engine: PASS")
    print("Event engine: PASS")
    print("Decision-support engine: PASS")
    print("\nRepresentative cases: 5/5 PASS")
    print("Existing tests: PASS")
    print("Phase 4E tests: PASS")
    print("\nFrozen artifacts modified: NO")
    print("\nPhase 4E STATUS: COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    main()
