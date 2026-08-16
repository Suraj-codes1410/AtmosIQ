"""
AtmosIQ Phase 7C CLI Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml.src.modeling.phase7c import Phase7CRunner, ValidationConfigPhase7C


def main():
    config = ValidationConfigPhase7C()
    runner = Phase7CRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
