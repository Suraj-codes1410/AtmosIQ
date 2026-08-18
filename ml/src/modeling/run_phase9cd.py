"""
AtmosIQ Phase 9C–9D CLI Entrypoint.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase9cd import Phase9CDRunner, Phase9CDConfig


def main():
    config = Phase9CDConfig()
    runner = Phase9CDRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
