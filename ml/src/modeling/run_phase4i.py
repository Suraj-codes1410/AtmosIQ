import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4i.runner import Phase4IRunner

if __name__ == "__main__":
    runner = Phase4IRunner()
    runner.run()
