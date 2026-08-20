"""
AtmosIQ Phase 10C CLI Entrypoint.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase10c import Phase10CRunner, Phase10CConfig


def main():
    config = Phase10CConfig()
    runner = Phase10CRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
