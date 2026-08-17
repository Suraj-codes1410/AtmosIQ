"""
AtmosIQ Phase 8F CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase8f import Phase8FRunner, Phase8FConfig


def main():
    config = Phase8FConfig()
    runner = Phase8FRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
