"""
AtmosIQ Phase 8D CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase8d import Phase8DRunner, CalibrationConfigPhase8D


def main():
    config = CalibrationConfigPhase8D()
    runner = Phase8DRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
