"""
AtmosIQ Phase 8E CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase8e import Phase8ERunner, Phase8EConfig


def main():
    config = Phase8EConfig()
    runner = Phase8ERunner(config)
    runner.run()


if __name__ == "__main__":
    main()
