"""
AtmosIQ Phase 8C CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase8c import Phase8CRunner, ReleaseConfigPhase8C


def main():
    config = ReleaseConfigPhase8C()
    runner = Phase8CRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
