#!/usr/bin/env python3
"""
AtmosIQ Phase 11B CLI runner.
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.src.modeling.phase11b.runner import Phase11BRunner

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    runner = Phase11BRunner(root_dir=ROOT)
    res = runner.run()
    sys.exit(0 if "ESTABLISHED" in res.get("final_decision", "") else 1)
