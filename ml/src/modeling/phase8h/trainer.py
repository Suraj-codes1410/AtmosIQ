"""
AtmosIQ Phase 8H: Deep-Learning Trainer & Checkpoint Engine.
"""

from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import json
import hashlib
import numpy as np
import logging

from .models import BasePhase8HModel
from .dataset import Phase8HSequenceDataset, Phase8HDataLoader

logger = logging.getLogger(__name__)


class Phase8HTrainer:
    """Orchestrates controlled training smoke tests, gradient audits, and checkpoint verification."""

    def __init__(self, model: BasePhase8HModel, lr: float = 0.001, seed: int = 42):
        self.model = model
        self.lr = lr
        self.seed = seed
        self.loss_history: List[float] = []

    def train_smoke(
        self,
        train_loader: Phase8HDataLoader,
        epochs: int = 5,
        checkpoint_path: Optional[Path] = None,
        corpus_sha: str = "",
        contract_version: str = "v1.1.0"
    ) -> Dict[str, Any]:
        """Runs controlled training smoke test and performs gradient/checkpoint audit."""
        initial_params = {k: v.copy() for k, v in self.model.params.items()}
        initial_loss = None

        for epoch in range(epochs):
            epoch_losses = []
            for X_b, y_b in train_loader:
                loss, grads = self.model.compute_loss_and_backward(X_b, y_b)
                if initial_loss is None:
                    initial_loss = loss
                epoch_losses.append(loss)
                self.model.optimizer_step(lr=self.lr)

            mean_epoch_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_epoch_loss)

        final_loss = self.loss_history[-1] if self.loss_history else 0.0
        grad_stats = self.model.get_gradient_stats()

        # Compute parameter delta
        param_deltas = [
            float(np.linalg.norm(self.model.params[k] - initial_params[k]))
            for k in self.model.params
        ]
        total_param_delta = float(sum(param_deltas))

        # Checkpoint round-trip test
        chk_summary = {}
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(checkpoint_path, corpus_sha=corpus_sha, contract_version=contract_version)
            
            # Predict before reload
            X_sample = next(iter(train_loader))[0][:10]
            preds_before = self.model.forward(X_sample)

            # Reload into fresh instance
            model_class = self.model.__class__
            reloaded_model = model_class(
                window_size=self.model.window_size,
                feature_dim=self.model.feature_dim,
                seed=self.model.seed
            )
            reloaded_meta = self.load_checkpoint(checkpoint_path, reloaded_model)
            preds_after = reloaded_model.forward(X_sample)

            inference_delta = float(np.max(np.abs(preds_before - preds_after)))
            chk_summary = {
                "checkpoint_saved": True,
                "checkpoint_file": str(checkpoint_path.name),
                "checkpoint_sha256": self.compute_file_sha256(checkpoint_path),
                "inference_delta": inference_delta,
                "round_trip_pass": bool(inference_delta <= 1e-9),
            }

        return {
            "model_name": self.model.name,
            "seed": self.seed,
            "epochs": epochs,
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "loss_decreased": bool(final_loss < initial_loss),
            "total_grad_norm": grad_stats["total_grad_norm"],
            "max_grad": grad_stats["max_grad"],
            "grad_nan_inf_free": bool(not grad_stats["has_nan"] and not grad_stats["has_inf"]),
            "total_param_delta": total_param_delta,
            "checkpoint_summary": chk_summary,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluates model performance metrics (MAE, RMSE, R²)."""
        y_pred = self.model.forward(X)
        mae = float(np.mean(np.abs(y_pred - y)))
        rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(np.std(y_pred)),
        }

    def save_checkpoint(self, path: Path, corpus_sha: str = "", contract_version: str = "v1.1.0"):
        """Saves structured checkpoint dictionary with provenance metadata."""
        ckpt = {
            "model_state": self.model.state_dict(),
            "metadata": {
                "architecture": self.model.name,
                "window_size": self.model.window_size,
                "feature_dim": self.model.feature_dim,
                "seed": self.seed,
                "corpus_sha256": corpus_sha,
                "contract_version": contract_version,
                "loss_history": self.loss_history,
            }
        }
        with open(path, "w") as f:
            json.dump(ckpt, f, indent=4)

    def load_checkpoint(self, path: Path, model: BasePhase8HModel) -> Dict[str, Any]:
        """Loads checkpoint and populates model weights."""
        with open(path) as f:
            ckpt = json.load(f)
        model.load_state_dict(ckpt["model_state"])
        return ckpt.get("metadata", {})

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
