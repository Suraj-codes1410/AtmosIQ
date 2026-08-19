"""
AtmosIQ Phase 10 + Phase 10A CLI Entrypoint.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase10 import Phase10Runner, Phase10Config


def main():
    config = Phase10Config()
    runner = Phase10Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
