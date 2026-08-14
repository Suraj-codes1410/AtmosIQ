#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4h.runner import Phase4HRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtmosIQ Phase 4H Dataset v3 Model Evaluation & Selection Runner")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna tuning trials per study")
    args = parser.parse_args()

    runner = Phase4HRunner()
    runner.run(optuna_trials=args.trials)
