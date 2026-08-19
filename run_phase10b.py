"""
AtmosIQ Phase 10B CLI Entrypoint in Root Directory.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase10b import Phase10BRunner, Phase10BConfig


def main():
    config = Phase10BConfig()
    runner = Phase10BRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
