"""
AtmosIQ Phase 10E CLI Entrypoint.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase10e import Phase10ERunner, Phase10EConfig


def main():
    config = Phase10EConfig()
    runner = Phase10ERunner(config)
    runner.run()


if __name__ == "__main__":
    main()
