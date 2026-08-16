import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase7b import Phase7BRunner, SyntheticConfigPhase7B


def main():
    config = SyntheticConfigPhase7B()
    runner = Phase7BRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
