"""
AtmosIQ Phase 8C: Extreme-Tail Governance Engine.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ExtremeTailGovernanceEngine:
    """Audits and filters synthetic extreme episodes against physical risk constraints."""

    def __init__(
        self,
        extreme_pm25_threshold: float = 250.0,
        vi_threshold: float = 4500.0,
        precipitation_threshold: float = 2.0
    ):
        self.extreme_pm25_threshold = extreme_pm25_threshold
        self.vi_threshold = vi_threshold
        self.precipitation_threshold = precipitation_threshold

    def audit_and_filter_corpus(self, df_corpus: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Audits every observation and trajectory in the corpus.
        Returns (df_compliant, df_governance_audit, summary_stats).
        """
        records = []
        rejected_trajectory_ids = set()

        # Group by trajectory
        for traj_id, df_traj in df_corpus.groupby("trajectory_id"):
            has_violation = False
            for _, row in df_traj.iterrows():
                pm = float(row["pm25"])
                vi = float(row.get("ventilation_index_1d", 0.0))
                rain = float(row.get("rainfall_1d", 0.0))
                date_str = str(row.get("synthetic_date", row.get("date", "UNKNOWN")))

                is_extreme = (pm >= self.extreme_pm25_threshold)
                incoherent_vi = is_extreme and (vi > self.vi_threshold)
                incoherent_rain = is_extreme and (rain > self.precipitation_threshold)

                if incoherent_vi or incoherent_rain:
                    has_violation = True
                    reason_parts = []
                    if incoherent_vi: reason_parts.append(f"VI ({vi:.1f}) > {self.vi_threshold}")
                    if incoherent_rain: reason_parts.append(f"Rain ({rain:.1f}mm) > {self.precipitation_threshold}mm")
                    reason = " & ".join(reason_parts)
                    decision = "REJECT"
                else:
                    reason = "NONE"
                    decision = "ACCEPT"

                records.append({
                    "trajectory_id": traj_id,
                    "synthetic_date": date_str,
                    "pm25": pm,
                    "ventilation_index_1d": vi,
                    "rainfall_1d": rain,
                    "filter_decision": decision,
                    "rejection_reason": reason,
                })

            if has_violation:
                rejected_trajectory_ids.add(traj_id)

        df_audit = pd.DataFrame(records)
        df_compliant = df_corpus[~df_corpus["trajectory_id"].isin(rejected_trajectory_ids)].copy()

        total_trajs = df_corpus["trajectory_id"].nunique()
        accepted_trajs = df_compliant["trajectory_id"].nunique()
        rejected_trajs = len(rejected_trajectory_ids)

        summary = {
            "total_candidate_trajectories": total_trajs,
            "accepted_trajectories": accepted_trajs,
            "rejected_trajectories": rejected_trajs,
            "rejection_rate_pct": float(rejected_trajs / total_trajs * 100.0) if total_trajs > 0 else 0.0,
            "total_candidate_observations": len(df_corpus),
            "accepted_observations": len(df_compliant),
            "rejected_observations": len(df_corpus) - len(df_compliant),
        }

        logger.info(
            f"Extreme-Tail Governance audit: {accepted_trajs}/{total_trajs} trajectories accepted "
            f"({len(df_compliant)}/{len(df_corpus)} observations compliant)."
        )

        return df_compliant, df_audit, summary
