"""
AtmosIQ Phase 7B: 4-State Seasonal Regime Markov Model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class RegimeMarkovModel:
    REGIMES = ["Low", "Moderate", "High", "Extreme"]
    SEASONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

    def __init__(self, random_state: np.random.RandomState):
        self.rng = random_state
        self.transition_matrices: Dict[str, np.ndarray] = {}
        self.initial_probabilities: Dict[str, np.ndarray] = {}

    def fit_from_training_data(self, df_train: pd.DataFrame):
        """Fit empirical seasonal transition matrices strictly on training partition."""
        df = df_train.copy().sort_values("date").reset_index(drop=True)
        
        # Ensure season and regime columns exist
        if "pollution_regime" not in df.columns:
            def classify_regime(pm):
                if pm < 60.0: return "Low"
                if pm < 120.0: return "Moderate"
                if pm < 250.0: return "High"
                return "Extreme"
            df["pollution_regime"] = df["pm25"].apply(classify_regime)

        if "season" not in df.columns:
            def classify_season(m):
                if m in [12, 1, 2]: return "Winter"
                if m in [3, 4, 5]: return "Summer"
                if m in [6, 7, 8, 9]: return "Monsoon"
                return "Post-Monsoon"
            df["month"] = pd.to_datetime(df["date"]).dt.month
            df["season"] = df["month"].apply(classify_season)

        df["next_regime"] = df["pollution_regime"].shift(-1)
        df_valid = df.dropna(subset=["next_regime"])

        for season in self.SEASONS:
            sub = df_valid[df_valid["season"] == season]
            if len(sub) == 0:
                # Default smoothing
                T = np.ones((4, 4)) / 4.0
                init_p = np.ones(4) / 4.0
            else:
                counts = np.zeros((4, 4))
                for _, row in sub.iterrows():
                    i = self.REGIMES.index(row["pollution_regime"])
                    j = self.REGIMES.index(row["next_regime"])
                    counts[i, j] += 1
                
                # Add Laplace smoothing to prevent zero transition traps
                counts += 0.5
                T = counts / counts.sum(axis=1, keepdims=True)

                init_counts = np.zeros(4)
                for r in sub["pollution_regime"]:
                    init_counts[self.REGIMES.index(r)] += 1
                init_counts += 0.5
                init_p = init_counts / init_counts.sum()

            self.transition_matrices[season] = T
            self.initial_probabilities[season] = init_p

    def sample_regime_sequence(self, length: int, season: str, initial_regime: str = None) -> List[str]:
        """Sample a sequence of regimes using Markov chain."""
        T = self.transition_matrices.get(season, np.eye(4))
        init_p = self.initial_probabilities.get(season, np.ones(4) / 4.0)

        sequence = []
        if initial_regime and initial_regime in self.REGIMES:
            curr_idx = self.REGIMES.index(initial_regime)
        else:
            curr_idx = self.rng.choice(4, p=init_p)

        sequence.append(self.REGIMES[curr_idx])

        for _ in range(1, length):
            curr_idx = self.rng.choice(4, p=T[curr_idx])
            sequence.append(self.REGIMES[curr_idx])

        return sequence
