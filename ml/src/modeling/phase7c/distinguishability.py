"""
AtmosIQ Phase 7C: Real vs Synthetic Classifier Distinguishability Test (Workstream G).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score


class RealVsSyntheticClassifier:
    """Trains a discriminator on development real vs synthetic data to quantify artifacts."""

    def __init__(self, feature_registry: List[str], random_seed: int = 42):
        self.feature_registry = list(feature_registry)
        self.random_seed = random_seed

    def evaluate_distinguishability(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray]:
        common = [f for f in self.feature_registry if f in df_real_train.columns and f in df_synthetic.columns]

        X_real = df_real_train[common].copy()
        X_real["label"] = 0

        X_synth = df_synthetic[common].copy()
        X_synth["label"] = 1

        df_combined = pd.concat([X_real, X_synth], ignore_index=True)
        X = df_combined[common].values
        y = df_combined["label"].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        oof_preds = np.zeros(len(y))
        feature_importances = np.zeros(len(common))

        for train_idx, val_idx in skf.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=4,
                random_state=self.random_seed,
                n_jobs=-1
            )
            clf.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
            feature_importances += clf.feature_importances_ / 5.0

        roc_auc = float(roc_auc_score(y, oof_preds))
        pr_auc = float(average_precision_score(y, oof_preds))
        binary_preds = (oof_preds >= 0.5).astype(int)

        acc = float(accuracy_score(y, binary_preds))
        prec = float(precision_score(y, binary_preds))
        rec = float(recall_score(y, binary_preds))
        f1 = float(f1_score(y, binary_preds))

        metrics_df = pd.DataFrame([{
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "sample_count_real": len(X_real),
            "sample_count_synth": len(X_synth),
        }])

        df_imp = pd.DataFrame({
            "feature_name": common,
            "importance": feature_importances,
        }).sort_values("importance", ascending=False)

        summary = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "f1_score": f1,
            "top_discriminating_feature": df_imp.iloc[0]["feature_name"],
            "distinguishability_status": "EXCELLENT_REALISM" if roc_auc <= 0.70 else ("ACCEPTABLE_ARTIFACTS" if roc_auc <= 0.88 else "DETECTABLE_ARTIFACTS"),
        }

        return metrics_df, df_imp, summary, y, oof_preds
