"""
AtmosIQ Phase 9: Model Selection & Ranking Engine.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class Phase9ModelSelector:
    """Ranks and selects Phase 9 deep-learning candidate models based on validation evidence."""

    def __init__(self, manifests_dir: Path):
        self.manifests_dir = manifests_dir

    def rank_models(self, val_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Ranks candidate configurations using multi-objective validation criteria."""
        df = pd.DataFrame(val_results)
        # Composite selection score: Lower is better (val_mae * 0.4 + extreme_mae * 0.3 + val_rmse * 0.3)
        df["selection_score"] = (
            df["val_mae"] * 0.40 +
            df["val_extreme_mae"] * 0.30 +
            df["val_rmse"] * 0.30
        )
        df = df.sort_values(by="selection_score", ascending=True).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def select_winning_candidate(
        self,
        ranked_df: pd.DataFrame,
        test_eval_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formally selects top research candidate and exports selection manifest."""
        winner = ranked_df.iloc[0].to_dict()

        selection_manifest = {
            "selection_status": "APPROVED_RESEARCH_CANDIDATE",
            "selected_architecture": winner["architecture"],
            "selected_corpus": winner["corpus"],
            "selected_augmentation_ratio": winner["augmentation_ratio"],
            "selected_seed": winner["seed"],
            "validation_metrics": {
                "val_mae": winner["val_mae"],
                "val_rmse": winner["val_rmse"],
                "val_r2": winner["val_r2"],
                "val_extreme_mae": winner["val_extreme_mae"],
                "selection_score": winner["selection_score"],
            },
            "locked_test_fold_metrics": test_eval_results,
            "candidate_designation": "AtmosIQ_Phase9_Temporal_Deep_Learning_Candidate_v1.0.0",
            "governance_note": "Selected model is certified as a research candidate. Production deployment requires separate operational admission gate.",
        }

        with open(self.manifests_dir / "phase9_model_selection.json", "w") as f:
            json.dump(selection_manifest, f, indent=4)

        return selection_manifest
