"""
AtmosIQ Phase 8D: Trajectory-Level Calibration Strategy Engine.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class CalibrationStrategyEngine:
    """Implements 8 controlled trajectory-level calibration strategies."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)
        self.scaler = StandardScaler()
        self.nn_dev = None
        self.real_acf = None
        self.real_corr = None
        self.real_dev_df = None

    def fit_from_development_data(self, df_real_dev: pd.DataFrame):
        """Fits historical reference distributions and moments on 2020-2021 training data."""
        self.real_dev_df = df_real_dev.copy()
        common = [f for f in self.feature_registry if f in df_real_dev.columns]

        # 1. Fit feature space standardizer and nearest neighbors for OOD
        X_real = df_real_dev[common].values
        X_scaled = self.scaler.fit_transform(X_real)
        self.nn_dev = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
        self.nn_dev.fit(X_scaled)
        dists, _ = self.nn_dev.kneighbors(X_scaled)
        self.baseline_p95_distance = float(np.percentile(dists[:, 1], 95))

        # 2. Fit real reference correlation matrix
        self.real_corr = df_real_dev[common].corr(method="pearson").fillna(0.0).values

        # 3. Fit real reference ACF (lags 1-30)
        def compute_acf(s, max_lag=30):
            s = s - np.mean(s)
            var = np.var(s)
            if var == 0: return np.ones(max_lag)
            return [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in range(1, max_lag + 1)]

        self.real_acf = compute_acf(df_real_dev["pm25"].dropna().values, 30)
        logger.info("CalibrationStrategyEngine successfully calibrated on historical development data.")

    def compute_trajectory_metrics(self, df_traj: pd.DataFrame) -> Dict[str, float]:
        """Calculates trajectory-level statistics for scoring and calibration selection."""
        common = [f for f in self.feature_registry if f in df_traj.columns]
        pm_vals = df_traj["pm25"].values
        real_pm = self.real_dev_df["pm25"].values

        # W1 on PM2.5 and primary features
        w1_pm = float(wasserstein_distance(pm_vals / max(np.std(real_pm), 1.0), real_pm / max(np.std(real_pm), 1.0)))

        # Trajectory ACF
        def compute_acf(s, max_lag=7):
            s = s - np.mean(s)
            var = np.var(s)
            if var <= 1e-6 or len(s) <= max_lag + 1: return [1.0] * max_lag
            return [float(np.corrcoef(s[:-k], s[k:])[0, 1]) if not np.isnan(np.corrcoef(s[:-k], s[k:])[0, 1]) else 0.0 for k in range(1, max_lag + 1)]

        traj_acf = compute_acf(pm_vals, 7)
        acf_err = float(np.mean([abs(traj_acf[i] - self.real_acf[i]) for i in range(len(traj_acf))]))

        # Correlation distance
        traj_corr = df_traj[common].corr(method="pearson").fillna(0.0).values
        d = len(common)
        corr_dist = float(norm(self.real_corr - traj_corr, "fro") / d) if d > 0 else 0.0

        # OOD score
        X_t_scaled = self.scaler.transform(df_traj[common].values)
        dists, _ = self.nn_dev.kneighbors(X_t_scaled, n_neighbors=1)
        mean_ood_dist = float(np.mean(dists[:, 0]))
        p95_ood_dist = float(np.percentile(dists[:, 0], 95))

        # Extreme count
        ext_count = int((df_traj["pm25"] >= 250.0).sum())

        return {
            "w1_pm": w1_pm,
            "acf_err": acf_err,
            "corr_dist": corr_dist,
            "mean_ood_dist": mean_ood_dist,
            "p95_ood_dist": p95_ood_dist,
            "ext_count": ext_count,
        }

    def apply_candidate_calibration(
        self,
        df_corpus: pd.DataFrame,
        candidate_id: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Applies candidate calibration filter and returns calibrated subset dataframe and stats."""
        trajs = list(df_corpus.groupby("trajectory_id"))
        total_trajs = len(trajs)

        # Precompute metrics per trajectory
        traj_scores = {}
        for traj_id, df_t in trajs:
            traj_scores[traj_id] = self.compute_trajectory_metrics(df_t)

        selected_traj_ids = []

        if candidate_id == "CAL-00":
            # Baseline: accept all
            selected_traj_ids = [t_id for t_id, _ in trajs]

        elif candidate_id == "CAL-01":
            # Distribution: select top 80% trajectories with lowest W1 distance
            sorted_by_w1 = sorted(traj_scores.keys(), key=lambda k: traj_scores[k]["w1_pm"])
            k_sel = int(len(sorted_by_w1) * 0.80)
            selected_traj_ids = sorted_by_w1[:k_sel]

        elif candidate_id == "CAL-02":
            # Regime: balance seasons and pollution regimes uniformly
            season_buckets = {}
            for t_id, df_t in trajs:
                s = df_t["season"].iloc[0] if "season" in df_t else "Winter"
                season_buckets.setdefault(s, []).append(t_id)
            for s, t_ids in season_buckets.items():
                selected_traj_ids.extend(t_ids[:int(len(t_ids) * 0.80)])

        elif candidate_id == "CAL-03":
            # Temporal: reject trajectories with top 20% ACF error
            sorted_by_acf = sorted(traj_scores.keys(), key=lambda k: traj_scores[k]["acf_err"])
            k_sel = int(len(sorted_by_acf) * 0.80)
            selected_traj_ids = sorted_by_acf[:k_sel]

        elif candidate_id == "CAL-04":
            # Multivariate: select trajectories with lowest Frobenius correlation distance
            sorted_by_corr = sorted(traj_scores.keys(), key=lambda k: traj_scores[k]["corr_dist"])
            k_sel = int(len(sorted_by_corr) * 0.80)
            selected_traj_ids = sorted_by_corr[:k_sel]

        elif candidate_id == "CAL-05":
            # OOD-aware: prune trajectories with extreme OOD distance (> 90th percentile)
            sorted_by_ood = sorted(traj_scores.keys(), key=lambda k: traj_scores[k]["p95_ood_dist"])
            k_sel = int(len(sorted_by_ood) * 0.85)
            selected_traj_ids = sorted_by_ood[:k_sel]

        elif candidate_id == "CAL-06":
            # Extreme-tail: preserve extreme trajectories only if VI <= 4000 and rain <= 1.0
            for t_id, df_t in trajs:
                ext_mask = (df_t["pm25"] >= 250.0)
                if ext_mask.any():
                    coherent = (df_t.loc[ext_mask, "ventilation_index_1d"] <= 4000.0) & (df_t.loc[ext_mask, "rainfall_1d"] <= 1.0)
                    if coherent.all():
                        selected_traj_ids.append(t_id)
                else:
                    selected_traj_ids.append(t_id)

        elif candidate_id == "CAL-07":
            # Combined multi-objective: composite score = 0.35*W1 + 0.30*ACF + 0.20*Corr + 0.15*OOD
            w1_vals = np.array([traj_scores[k]["w1_pm"] for k in traj_scores])
            acf_vals = np.array([traj_scores[k]["acf_err"] for k in traj_scores])
            corr_vals = np.array([traj_scores[k]["corr_dist"] for k in traj_scores])
            ood_vals = np.array([traj_scores[k]["p95_ood_dist"] for k in traj_scores])

            w1_norm = (w1_vals - np.min(w1_vals)) / (np.ptp(w1_vals) + 1e-6)
            acf_norm = (acf_vals - np.min(acf_vals)) / (np.ptp(acf_vals) + 1e-6)
            corr_norm = (corr_vals - np.min(corr_vals)) / (np.ptp(corr_vals) + 1e-6)
            ood_norm = (ood_vals - np.min(ood_vals)) / (np.ptp(ood_vals) + 1e-6)

            composite = 0.35 * w1_norm + 0.30 * acf_norm + 0.20 * corr_norm + 0.15 * ood_norm
            scored_keys = list(traj_scores.keys())
            sorted_indices = np.argsort(composite)
            k_sel = int(len(scored_keys) * 0.80)
            selected_traj_ids = [scored_keys[i] for i in sorted_indices[:k_sel]]

        selected_set = set(selected_traj_ids)
        df_calibrated = df_corpus[df_corpus["trajectory_id"].isin(selected_set)].copy().reset_index(drop=True)

        stats = {
            "candidate_id": candidate_id,
            "total_candidate_trajectories": total_trajs,
            "accepted_trajectories": len(selected_set),
            "rejected_trajectories": total_trajs - len(selected_set),
            "acceptance_rate_pct": float(len(selected_set) / total_trajs * 100.0) if total_trajs > 0 else 0.0,
            "calibrated_observations": len(df_calibrated),
        }

        logger.info(
            f"Candidate {candidate_id}: {stats['accepted_trajectories']}/{total_trajs} trajectories accepted "
            f"({stats['calibrated_observations']} observations, Acceptance: {stats['acceptance_rate_pct']:.1f}%)."
        )

        return df_calibrated, stats
