import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase3g.phase3g_runner import MasterRunnerPhase3G

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtmosIQ Phase 3G Optuna Tuning Engine")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials per study")
    parser.add_argument("--model", type=str, default="all", help="Target model to tune (all, xgboost, random_forest, ridge, elasticnet)")
    args = parser.parse_args()

    runner = MasterRunnerPhase3G()
    runner.run(n_trials=args.trials, target_model=args.model)
