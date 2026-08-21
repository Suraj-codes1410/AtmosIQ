#!/usr/bin/env python3
"""
AtmosIQ Phase 11A — CLI entry point.

Usage:
    python run_phase11a.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).parent

sys.path.insert(0, str(ROOT))

from ml.src.modeling.phase11a.runner import Phase11ARunner

if __name__ == "__main__":
    runner = Phase11ARunner(root_dir=ROOT)
    result = runner.run()
    decision = result.get("final_decision", "UNKNOWN")
    sys.exit(0 if "VALIDATED" in decision else 1)
