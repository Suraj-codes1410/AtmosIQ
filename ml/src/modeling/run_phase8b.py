"""
AtmosIQ Phase 8B CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase8b import Phase8BRunner, ScalingConfigPhase8B


def main():
    config = ScalingConfigPhase8B()
    runner = Phase8BRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
