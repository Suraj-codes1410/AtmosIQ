#!/usr/bin/env python3
"""
AtmosIQ Phase 11B — CLI Entrypoint.

Usage:
    python run_phase11b.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ml.src.modeling.phase11b.runner import Phase11BRunner

if __name__ == "__main__":
    runner = Phase11BRunner(root_dir=ROOT)
    results = runner.run()
    decision = results.get("final_decision", "UNKNOWN")
    sys.exit(0 if "ESTABLISHED" in decision else 1)
