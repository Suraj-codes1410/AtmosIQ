"""
AtmosIQ Phase 9 CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase9 import Phase9Runner, Phase9Config


def main():
    config = Phase9Config()
    runner = Phase9Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
