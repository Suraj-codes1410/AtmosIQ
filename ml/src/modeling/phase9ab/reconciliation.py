"""
AtmosIQ Phase 9A: Model Selection Reconciliation & Governance Resolution Engine.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class Phase9AReconciler:
    """Reconciles Phase 9 candidate model rankings with upstream governance constraints."""

    def __init__(self, benchmarks_dir: Path, manifests_dir: Path):
        self.benchmarks_dir = benchmarks_dir
        self.manifests_dir = manifests_dir

    def reconcile_candidates(self, p9_val_csv: Path, p9_multi_csv: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Reconstructs candidate performance and assigns governance-aware certification statuses."""
        df_val = pd.read_csv(p9_val_csv)
        df_multi = pd.read_csv(p9_multi_csv)

        # Aggregate across seeds for candidate-level evaluation
        grouped = df_val.groupby(["architecture", "augmentation_ratio"]).agg(
            val_mae_mean=("val_mae", "mean"),
            val_mae_std=("val_mae", "std"),
            val_rmse_mean=("val_rmse", "mean"),
            val_r2_mean=("val_r2", "mean"),
            val_pearson_mean=("val_pearson_r", "mean"),
            val_extreme_mae_mean=("val_extreme_mae", "mean"),
            val_extreme_rmse_mean=("val_extreme_rmse", "mean"),
        ).reset_index()

        reconciliation_records = []
        for idx, row in grouped.iterrows():
            arch = row["architecture"]
            aug = row["augmentation_ratio"]
            candidate_id = f"{arch}_aug{int(aug*100)}pct"
            corpus = "AtmosIQ_Synthetic_Calibrated_v0.1.0" if aug > 0 else "REAL_ONLY"

            # Selection composite score (MAE*0.4 + ExtremeMAE*0.3 + RMSE*0.3)
            score = row["val_mae_mean"] * 0.40 + row["val_extreme_mae_mean"] * 0.30 + row["val_rmse_mean"] * 0.30

            # Governance classification logic
            if aug == 0.0:
                gov_status = "APPROVED_BASELINE"
                cert_status = "PRODUCTION_ELIGIBLE"
                rejection_reason = "NONE"
                promotion_reason = "Pure empirical historical baseline without synthetic augmentation."
            elif aug == 0.10:
                gov_status = "APPROVED_CONSERVATIVE_AUGMENTATION"
                cert_status = "PRODUCTION_ELIGIBLE"
                rejection_reason = "NONE"
                promotion_reason = "Conservative 10% augmentation compliant with Phase 8G envelope."
            elif aug == 0.25:
                gov_status = "APPROVED_PRODUCTION_DEFAULT"
                cert_status = "PRODUCTION_ELIGIBLE"
                rejection_reason = "NONE"
                promotion_reason = "Primary approved production default configuration established in Phase 8E/8G."
            elif aug == 0.50:
                gov_status = "CONTROLLED_STRESS_TEST_UPPER_BOUND"
                cert_status = "CERTIFIED_RESEARCH_CANDIDATE"
                rejection_reason = "Requires explicit operational admission gate before direct production deployment."
                promotion_reason = "Top-performing empirical stress-test research candidate on extreme events."
            else:
                gov_status = "PROHIBITED"
                cert_status = "REJECTED"
                rejection_reason = "Exceeds 50% upper bound or violates augmentation firewall."
                promotion_reason = "NONE"

            # Stability indices
            temporal_stability = "HIGH" if row["val_mae_std"] < 0.20 else "MODERATE"
            seasonal_stability = "HIGH" if row["val_extreme_mae_mean"] < 50.0 else "MODERATE"
            regime_stability = "HIGH" if row["val_r2_mean"] > 0.60 else "MODERATE"

            reconciliation_records.append({
                "candidate_id": candidate_id,
                "architecture": arch,
                "augmentation_ratio": aug,
                "corpus": corpus,
                "validation_mae": float(row["val_mae_mean"]),
                "validation_rmse": float(row["val_rmse_mean"]),
                "validation_r2": float(row["val_r2_mean"]),
                "pearson_r": float(row["val_pearson_mean"]),
                "extreme_mae": float(row["val_extreme_mae_mean"]),
                "seed_mean": float(row["val_mae_mean"]),
                "seed_std": float(row["val_mae_std"]),
                "temporal_stability": temporal_stability,
                "seasonal_stability": seasonal_stability,
                "regime_stability": regime_stability,
                "governance_status": gov_status,
                "selection_score": float(score),
                "certification_status": cert_status,
                "rejection_reason": rejection_reason,
                "promotion_reason": promotion_reason,
            })

        df_recon = pd.DataFrame(reconciliation_records)
        df_recon = df_recon.sort_values(by="selection_score", ascending=True).reset_index(drop=True)

        # Explicit decision resolution
        research_winner = df_recon[df_recon["certification_status"] == "CERTIFIED_RESEARCH_CANDIDATE"].iloc[0].to_dict()
        production_winner = df_recon[df_recon["certification_status"] == "PRODUCTION_ELIGIBLE"].iloc[0].to_dict()

        decision = {
            "phase": "Phase 9A",
            "reconciliation_status": "GOVERNANCE_RECONCILED",
            "governance_rule": "50% augmentation is certified strictly as a research candidate; 25% augmentation remains the primary production-eligible candidate.",
            "selected_research_candidate": {
                "candidate_id": research_winner["candidate_id"],
                "architecture": research_winner["architecture"],
                "augmentation_ratio": research_winner["augmentation_ratio"],
                "corpus": research_winner["corpus"],
                "selection_score": research_winner["selection_score"],
                "certification_status": research_winner["certification_status"],
                "governance_role": "RESEARCH_CANDIDATE_STRESS_TEST",
            },
            "selected_production_eligible_candidate": {
                "candidate_id": production_winner["candidate_id"],
                "architecture": production_winner["architecture"],
                "augmentation_ratio": production_winner["augmentation_ratio"],
                "corpus": production_winner["corpus"],
                "selection_score": production_winner["selection_score"],
                "certification_status": production_winner["certification_status"],
                "governance_role": "APPROVED_PRODUCTION_DEFAULT",
            },
            "total_candidates_evaluated": len(df_recon),
            "prohibited_candidates_count": 0,
        }

        return df_recon, decision
