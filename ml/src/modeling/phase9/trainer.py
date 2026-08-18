"""
AtmosIQ Phase 9: Deep-Learning Model Trainer & Checkpoint Engine.
"""

from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import json
import hashlib
import numpy as np
import logging

from .models import BasePhase9Model
from .dataset import Phase9SequenceDataset, Phase9DataLoader

logger = logging.getLogger(__name__)


class Phase9Trainer:
    """Orchestrates Phase 9 deep-learning model training with validation monitoring and checkpointing."""

    def __init__(self, model: BasePhase9Model, lr: float = 0.001, seed: int = 42):
        self.model = model
        self.lr = lr
        self.seed = seed
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_params: Dict[str, np.ndarray] = {}

    def fit(
        self,
        train_loader: Phase9DataLoader,
        val_loader: Optional[Phase9DataLoader] = None,
        epochs: int = 25,
        checkpoint_path: Optional[Path] = None,
        corpus_sha: str = "",
        contract_version: str = "v1.1.0"
    ) -> Dict[str, Any]:
        """Trains the model across specified epochs with validation monitoring."""
        for epoch in range(1, epochs + 1):
            epoch_train_losses = []
            for X_b, y_b in train_loader:
                loss, grads = self.model.compute_loss_and_backward(X_b, y_b)
                epoch_train_losses.append(loss)
                self.model.optimizer_step(lr=self.lr)

            mean_train_loss = float(np.mean(epoch_train_losses))
            self.train_losses.append(mean_train_loss)

            # Validation step
            if val_loader is not None and len(val_loader) > 0:
                epoch_val_losses = []
                for X_v, y_v in val_loader:
                    y_v_pred = self.model.forward(X_v)
                    v_loss = float(np.mean((y_v_pred - y_v) ** 2))
                    epoch_val_losses.append(v_loss)
                mean_val_loss = float(np.mean(epoch_val_losses))
                self.val_losses.append(mean_val_loss)

                if mean_val_loss < self.best_val_loss:
                    self.best_val_loss = mean_val_loss
                    self.best_epoch = epoch
                    self.best_params = {k: v.copy() for k, v in self.model.params.items()}
            else:
                self.val_losses.append(mean_train_loss)
                if mean_train_loss < self.best_val_loss:
                    self.best_val_loss = mean_train_loss
                    self.best_epoch = epoch
                    self.best_params = {k: v.copy() for k, v in self.model.params.items()}

        # Restore best weights
        if self.best_params:
            self.model.params = {k: v.copy() for k, v in self.best_params.items()}

        grad_stats = self.model.get_gradient_stats()

        # Save checkpoint if requested
        chk_summary = {}
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(checkpoint_path, corpus_sha=corpus_sha, contract_version=contract_version)
            chk_summary = {
                "checkpoint_saved": True,
                "checkpoint_file": str(checkpoint_path.name),
                "checkpoint_sha256": self.compute_file_sha256(checkpoint_path),
            }

        return {
            "model_name": self.model.name,
            "seed": self.seed,
            "epochs": epochs,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.train_losses[-1],
            "best_val_loss": self.best_val_loss,
            "total_grad_norm": grad_stats["total_grad_norm"],
            "max_grad": grad_stats["max_grad"],
            "grad_nan_inf_free": bool(not grad_stats["has_nan"] and not grad_stats["has_inf"]),
            "checkpoint_summary": chk_summary,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.forward(X)

    def save_checkpoint(self, path: Path, corpus_sha: str = "", contract_version: str = "v1.1.0"):
        ckpt = {
            "model_state": self.model.state_dict(),
            "metadata": {
                "architecture": self.model.name,
                "window_size": self.model.window_size,
                "feature_dim": self.model.feature_dim,
                "seed": self.seed,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "corpus_sha256": corpus_sha,
                "contract_version": contract_version,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }
        }
        with open(path, "w") as f:
            json.dump(ckpt, f, indent=4)

    def load_checkpoint(self, path: Path, model: BasePhase9Model) -> Dict[str, Any]:
        with open(path) as f:
            ckpt = json.load(f)
        model.load_state_dict(ckpt["model_state"])
        return ckpt.get("metadata", {})

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
