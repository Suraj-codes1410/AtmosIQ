import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6f.config import DecisionSupportConfigPhase6F

logger = setup_logger("OODAdapterPhase6F")


class OODAdapterPhase6F:
    """
    Production Out-Of-Distribution (OOD) Adapter for Phase 6F.
    Evaluates standardized feature space distance against historical training distributions.
    """

    def __init__(
        self,
        config: DecisionSupportConfigPhase6F,
        dataset_path: str = "ml/data/modeling/v3/feature_dataset_frozen.csv",
        feature_names: Optional[List[str]] = None
    ):
        self.config = config
        self.df_train = pd.read_csv(Path(dataset_path))
        
        # Historical baseline: 2020-2021
        hist = self.df_train[pd.to_datetime(self.df_train['date']).dt.year.isin([2020, 2021])].copy()
        if len(hist) == 0:
            hist = self.df_train.copy()

        self.means = {}
        self.stds = {}
        for col in hist.columns:
            if pd.api.types.is_numeric_dtype(hist[col]):
                self.means[col] = float(hist[col].mean())
                self.stds[col] = float(hist[col].std()) + 1e-8

    def evaluate_ood(self, x_vec: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Calculates standardized distance score for input vector x.
        """
        if x_vec.ndim == 1:
            vals = x_vec
        else:
            vals = x_vec[0]

        z_scores = []
        for j, feat_name in enumerate(feature_names):
            m = self.means.get(feat_name, 0.0)
            s = self.stds.get(feat_name, 1.0)
            z = abs((float(vals[j]) - m) / s)
            z_scores.append((feat_name, z))

        z_scores.sort(key=lambda x: x[1], reverse=True)
        max_feat, max_z = z_scores[0]
        mean_z = float(np.mean([z for _, z in z_scores]))

        # OOD score definition: standardized distance metric
        ood_score = float(max(min(max_z / 2.0, 5.0), 1.0))

        if ood_score > self.config.ood_near_ood_threshold:
            status = "OOD"
            warning_msg = f"Elevated feature divergence detected on '{max_feat}' (z-score: {max_z:.2f})."
        elif ood_score > self.config.ood_in_distribution_threshold:
            status = "NEAR_OOD"
            warning_msg = f"Moderate distribution shift detected on '{max_feat}' (z-score: {max_z:.2f})."
        else:
            status = "IN_DISTRIBUTION"
            warning_msg = "Input features within normal historical distribution bounds."

        return {
            "ood_score": ood_score,
            "ood_status": status,
            "max_z_score": float(max_z),
            "mean_z_score": mean_z,
            "most_deviated_feature": max_feat,
            "warning_message": warning_msg
        }
