"""
AtmosIQ Phase 8G CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase8g import Phase8GRunner, Phase8GConfig


def main():
    config = Phase8GConfig()
    runner = Phase8GRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
