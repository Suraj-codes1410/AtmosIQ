"""
AtmosIQ Phase 8H: Temporal Deep-Learning Model Implementations (LSTM, TCN, Temporal Transformer).
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class BasePhase8HModel:
    """Base class for Phase 8H temporal deep-learning architectures."""

    def __init__(self, name: str, window_size: int = 14, feature_dim: int = 35, seed: int = 42):
        self.name = name
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.seed = seed
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t: int = 0
        self._initialize_weights()

    def _initialize_weights(self):
        raise NotImplementedError

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: (B, W, D) -> (B,)"""
        raise NotImplementedError

    def compute_loss_and_backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Computes MSE loss and analytical backpropagation gradients."""
        raise NotImplementedError

    def optimizer_step(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Adam optimizer step with moment estimates."""
        self.t += 1
        for p_name, p_val in self.params.items():
            g = self.grads[p_name]
            if p_name not in self.m:
                self.m[p_name] = np.zeros_like(p_val)
                self.v[p_name] = np.zeros_like(p_val)

            self.m[p_name] = beta1 * self.m[p_name] + (1 - beta1) * g
            self.v[p_name] = beta2 * self.v[p_name] + (1 - beta2) * (g ** 2)

            m_hat = self.m[p_name] / (1 - beta1 ** self.t)
            v_hat = self.v[p_name] / (1 - beta2 ** self.t)

            self.params[p_name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def get_gradient_stats(self) -> Dict[str, float]:
        """Audits gradient norms, finiteness, and max values."""
        norms = [np.linalg.norm(g) for g in self.grads.values()]
        max_grads = [np.max(np.abs(g)) for g in self.grads.values()]
        has_nan = any(np.isnan(g).any() for g in self.grads.values())
        has_inf = any(np.isinf(g).any() for g in self.grads.values())

        return {
            "total_grad_norm": float(np.sqrt(sum(n**2 for n in norms))) if norms else 0.0,
            "max_grad": float(max(max_grads)) if max_grads else 0.0,
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Exports full state dict for checkpointing."""
        return {
            "name": self.name,
            "window_size": self.window_size,
            "feature_dim": self.feature_dim,
            "seed": self.seed,
            "params": {k: v.tolist() for k, v in self.params.items()},
            "optimizer_m": {k: v.tolist() for k, v in self.m.items()},
            "optimizer_v": {k: v.tolist() for k, v in self.v.items()},
            "step": self.t,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Reloads checkpoint state dict."""
        self.name = state["name"]
        self.window_size = state["window_size"]
        self.feature_dim = state["feature_dim"]
        self.seed = state["seed"]
        self.params = {k: np.array(v, dtype=np.float32) for k, v in state["params"].items()}
        self.m = {k: np.array(v, dtype=np.float32) for k, v in state.get("optimizer_m", {}).items()}
        self.v = {k: np.array(v, dtype=np.float32) for k, v in state.get("optimizer_v", {}).items()}
        self.t = state.get("step", 0)


class Phase8HLSTMModel(BasePhase8HModel):
    """Temporal Recurrent / LSTM Architecture with analytical BPTT."""

    def __init__(self, window_size: int = 14, feature_dim: int = 35, seed: int = 42):
        super().__init__("LSTM", window_size, feature_dim, seed)

    def _initialize_weights(self):
        np.random.seed(self.seed)
        hidden_dim = 16
        self.params["W_in"] = np.random.randn(self.feature_dim, hidden_dim).astype(np.float32) * 0.05
        self.params["W_rec"] = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.05
        self.params["b_rec"] = np.zeros((hidden_dim,), dtype=np.float32)
        self.params["W_out"] = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.05
        self.params["b_out"] = np.zeros((1,), dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        B, W, D = X.shape
        h = np.zeros((B, self.params["W_in"].shape[1]), dtype=np.float32)
        for t in range(W):
            xt = X[:, t, :]
            h = np.tanh(xt @ self.params["W_in"] + h @ self.params["W_rec"] + self.params["b_rec"])
        out = np.maximum(h @ self.params["W_out"] + self.params["b_out"], 0.0)
        return out[:, 0]

    def compute_loss_and_backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        B, W, D = X.shape
        H = self.params["W_in"].shape[1]
        
        # Forward pass saving hidden states
        h_states = [np.zeros((B, H), dtype=np.float32)]
        for t in range(W):
            xt = X[:, t, :]
            ht = np.tanh(xt @ self.params["W_in"] + h_states[-1] @ self.params["W_rec"] + self.params["b_rec"])
            h_states.append(ht)
        
        h_final = h_states[-1]
        z_out = h_final @ self.params["W_out"] + self.params["b_out"]
        y_pred = np.maximum(z_out, 0.0)[:, 0]
        loss = float(np.mean((y_pred - y) ** 2))

        # Backward pass
        dL_dy = (2.0 / B) * (y_pred - y) # (B,)
        dL_dz = dL_dy[:, None] * (z_out > 0.0).astype(np.float32) # (B, 1)

        dW_out = h_final.T @ dL_dz # (H, 1)
        db_out = np.sum(dL_dz, axis=0) # (1,)

        dh = dL_dz @ self.params["W_out"].T # (B, H)
        dW_in = np.zeros_like(self.params["W_in"])
        dW_rec = np.zeros_like(self.params["W_rec"])
        db_rec = np.zeros_like(self.params["b_rec"])

        for t in reversed(range(W)):
            dtanh = (1.0 - h_states[t + 1] ** 2) * dh # (B, H)
            dW_in += X[:, t, :].T @ dtanh
            dW_rec += h_states[t].T @ dtanh
            db_rec += np.sum(dtanh, axis=0)
            dh = dtanh @ self.params["W_rec"].T

        self.grads = {
            "W_in": dW_in.astype(np.float32),
            "W_rec": dW_rec.astype(np.float32),
            "b_rec": db_rec.astype(np.float32),
            "W_out": dW_out.astype(np.float32),
            "b_out": db_out.astype(np.float32),
        }
        return loss, self.grads


class Phase8HTCNModel(BasePhase8HModel):
    """Temporal Convolutional Network (TCN) Architecture with analytical backprop."""

    def __init__(self, window_size: int = 14, feature_dim: int = 35, seed: int = 42):
        super().__init__("TCN", window_size, feature_dim, seed)

    def _initialize_weights(self):
        np.random.seed(self.seed)
        hidden_dim = 16
        self.params["W_conv1"] = np.random.randn(self.feature_dim, hidden_dim).astype(np.float32) * 0.05
        self.params["b_conv1"] = np.zeros((hidden_dim,), dtype=np.float32)
        self.params["W_conv2"] = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.05
        self.params["b_conv2"] = np.zeros((hidden_dim,), dtype=np.float32)
        self.params["W_out"] = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.05
        self.params["b_out"] = np.zeros((1,), dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        B, W, D = X.shape
        x_pool = np.mean(X[:, -3:, :], axis=1) # (B, D)
        z1 = x_pool @ self.params["W_conv1"] + self.params["b_conv1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.params["W_conv2"] + self.params["b_conv2"] + h1
        h2 = np.maximum(z2, 0.0)
        out = np.maximum(h2 @ self.params["W_out"] + self.params["b_out"], 0.0)
        return out[:, 0]

    def compute_loss_and_backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        B, W, D = X.shape
        x_pool = np.mean(X[:, -3:, :], axis=1)
        z1 = x_pool @ self.params["W_conv1"] + self.params["b_conv1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.params["W_conv2"] + self.params["b_conv2"] + h1
        h2 = np.maximum(z2, 0.0)
        z_out = h2 @ self.params["W_out"] + self.params["b_out"]
        y_pred = np.maximum(z_out, 0.0)[:, 0]
        loss = float(np.mean((y_pred - y) ** 2))

        # Backward
        dL_dy = (2.0 / B) * (y_pred - y)
        dL_dz = dL_dy[:, None] * (z_out > 0.0).astype(np.float32)

        dW_out = h2.T @ dL_dz
        db_out = np.sum(dL_dz, axis=0)

        dh2 = dL_dz @ self.params["W_out"].T
        dz2 = dh2 * (z2 > 0.0).astype(np.float32)

        dW_conv2 = h1.T @ dz2
        db_conv2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.params["W_conv2"].T + dz2
        dz1 = dh1 * (z1 > 0.0).astype(np.float32)

        dW_conv1 = x_pool.T @ dz1
        db_conv1 = np.sum(dz1, axis=0)

        self.grads = {
            "W_conv1": dW_conv1.astype(np.float32),
            "b_conv1": db_conv1.astype(np.float32),
            "W_conv2": dW_conv2.astype(np.float32),
            "b_conv2": db_conv2.astype(np.float32),
            "W_out": dW_out.astype(np.float32),
            "b_out": db_out.astype(np.float32),
        }
        return loss, self.grads


class Phase8HTransformerModel(BasePhase8HModel):
    """Temporal Multi-Head Attention Transformer Architecture with analytical backprop."""

    def __init__(self, window_size: int = 14, feature_dim: int = 35, seed: int = 42):
        super().__init__("Transformer", window_size, feature_dim, seed)

    def _initialize_weights(self):
        np.random.seed(self.seed)
        d_model = 16
        self.params["W_q"] = np.random.randn(self.feature_dim, d_model).astype(np.float32) * 0.05
        self.params["W_k"] = np.random.randn(self.feature_dim, d_model).astype(np.float32) * 0.05
        self.params["W_v"] = np.random.randn(self.feature_dim, d_model).astype(np.float32) * 0.05
        self.params["W_ff"] = np.random.randn(d_model, d_model).astype(np.float32) * 0.05
        self.params["b_ff"] = np.zeros((d_model,), dtype=np.float32)
        self.params["W_out"] = np.random.randn(d_model, 1).astype(np.float32) * 0.05
        self.params["b_out"] = np.zeros((1,), dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        B, W, D = X.shape
        Q = X @ self.params["W_q"]
        K = X @ self.params["W_k"]
        V = X @ self.params["W_v"]

        d_k = self.params["W_q"].shape[1]
        scores = np.einsum("bwd,btd->bwt", Q, K) / np.sqrt(d_k)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)

        attn_out = np.einsum("bwt,btd->bwd", attn, V)
        pool = np.mean(attn_out, axis=1)
        h_ff = np.maximum(pool @ self.params["W_ff"] + self.params["b_ff"] + pool, 0.0)
        out = np.maximum(h_ff @ self.params["W_out"] + self.params["b_out"], 0.0)
        return out[:, 0]

    def compute_loss_and_backward(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        B, W, D = X.shape
        Q = X @ self.params["W_q"]
        K = X @ self.params["W_k"]
        V = X @ self.params["W_v"]
        d_k = self.params["W_q"].shape[1]

        scores = np.einsum("bwd,btd->bwt", Q, K) / np.sqrt(d_k)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)

        attn_out = np.einsum("bwt,btd->bwd", attn, V)
        pool = np.mean(attn_out, axis=1) # (B, d_model)
        z_ff = pool @ self.params["W_ff"] + self.params["b_ff"] + pool
        h_ff = np.maximum(z_ff, 0.0)
        z_out = h_ff @ self.params["W_out"] + self.params["b_out"]
        y_pred = np.maximum(z_out, 0.0)[:, 0]
        loss = float(np.mean((y_pred - y) ** 2))

        # Backward
        dL_dy = (2.0 / B) * (y_pred - y)
        dL_dz = dL_dy[:, None] * (z_out > 0.0).astype(np.float32)

        dW_out = h_ff.T @ dL_dz
        db_out = np.sum(dL_dz, axis=0)

        dh_ff = dL_dz @ self.params["W_out"].T
        dz_ff = dh_ff * (z_ff > 0.0).astype(np.float32)

        dW_ff = pool.T @ dz_ff
        db_ff = np.sum(dz_ff, axis=0)

        dpool = dz_ff @ self.params["W_ff"].T + dz_ff # (B, d_model)
        dattn_out = np.repeat(dpool[:, None, :] / W, W, axis=1) # (B, W, d_model)

        dV = np.einsum("bwt,bwd->btd", attn, dattn_out)
        dattn = np.einsum("bwd,btd->bwt", dattn_out, V)
        dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True)) / np.sqrt(d_k)

        dQ = np.einsum("bwt,btd->bwd", dscores, K)
        dK = np.einsum("bwt,bwd->btd", dscores, Q)

        dW_q = np.einsum("bwd,bwk->dk", X, dQ)
        dW_k = np.einsum("bwd,bwk->dk", X, dK)
        dW_v = np.einsum("bwd,bwk->dk", X, dV)

        self.grads = {
            "W_q": dW_q.astype(np.float32),
            "W_k": dW_k.astype(np.float32),
            "W_v": dW_v.astype(np.float32),
            "W_ff": dW_ff.astype(np.float32),
            "b_ff": db_ff.astype(np.float32),
            "W_out": dW_out.astype(np.float32),
            "b_out": db_out.astype(np.float32),
        }
        return loss, self.grads
