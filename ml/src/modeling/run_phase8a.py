"""
AtmosIQ Phase 8A CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase8a import Phase8ARunner, GenerationConfigPhase8A


def main():
    config = GenerationConfigPhase8A(mode="PILOT")
    runner = Phase8ARunner(config)
    runner.run()


if __name__ == "__main__":
    main()
