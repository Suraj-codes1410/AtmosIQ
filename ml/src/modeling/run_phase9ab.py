"""
AtmosIQ Phase 9A–9B CLI Entrypoint.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase9ab import Phase9ABRunner, Phase9ABConfig


def main():
    config = Phase9ABConfig()
    runner = Phase9ABRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
