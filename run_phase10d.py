"""
AtmosIQ Phase 10D CLI Entrypoint in Root Directory.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase10d import Phase10DRunner, Phase10DConfig


def main():
    config = Phase10DConfig()
    runner = Phase10DRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
