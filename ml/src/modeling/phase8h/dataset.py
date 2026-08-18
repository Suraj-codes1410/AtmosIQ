"""
AtmosIQ Phase 8H: Temporal Sequence Dataset & Deterministic DataLoader.
"""

from typing import List, Dict, Any, Tuple, Iterator, Optional
import numpy as np
import pandas as pd


class Phase8HSequenceDataset:
    """Dataset container for temporal sequence tensors (N, W, D) and aligned targets (N,)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, provenance: Optional[List[Dict[str, Any]]] = None):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.provenance = provenance or []

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]


class Phase8HDataLoader:
    """Deterministic batch iterator for Phase 8H temporal sequences."""

    def __init__(self, dataset: Phase8HSequenceDataset, batch_size: int = 32, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.arange(len(dataset))

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices)

        for i in range(0, len(self.dataset), self.batch_size):
            batch_idx = self.indices[i : i + self.batch_size]
            yield self.dataset.X[batch_idx], self.dataset.y[batch_idx]
