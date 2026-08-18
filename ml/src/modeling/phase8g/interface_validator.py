"""
AtmosIQ Phase 8G: Deep-Learning Architecture Interface Validator.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Phase8GInterfaceValidator:
    """Verifies schema, tensor shapes, batch compatibility, and architecture neutrality for Phase 9."""

    def __init__(self, feature_registry: List[str]):
        self.feature_registry = list(feature_registry)

    def validate_training_tensors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 14
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verifies shape, data types, and numerical validity of the integrated training tensors."""
        checks = []

        # 1. Tensor 3D shape: (N, W, D)
        shape_valid = (len(X.shape) == 3 and X.shape[1] == window_size and X.shape[2] == len(self.feature_registry))
        checks.append({
            "check": "Tensor 3D Shape (N, W, D)",
            "expected_shape": f"(N, {window_size}, {len(self.feature_registry)})",
            "observed_shape": str(X.shape),
            "status": "PASS" if shape_valid else "FAIL",
        })

        # 2. Target 1D shape: (N,)
        target_shape_valid = (len(y.shape) == 1 and len(y) == len(X))
        checks.append({
            "check": "Target Aligned 1D Shape (N,)",
            "expected_shape": f"({len(X)},)",
            "observed_shape": str(y.shape),
            "status": "PASS" if target_shape_valid else "FAIL",
        })

        # 3. Data type
        dtype_valid = (X.dtype == np.float32 and y.dtype == np.float32)
        checks.append({
            "check": "Float32 Tensor Datatype",
            "observed_X_dtype": str(X.dtype),
            "observed_y_dtype": str(y.dtype),
            "status": "PASS" if dtype_valid else "FAIL",
        })

        # 4. Numerical Completeness (Zero NaN / Inf)
        nan_x = int(np.isnan(X).sum())
        nan_y = int(np.isnan(y).sum())
        inf_x = int(np.isinf(X).sum())
        inf_y = int(np.isinf(y).sum())
        num_clean = (nan_x == 0 and nan_y == 0 and inf_x == 0 and inf_y == 0)
        checks.append({
            "check": "Numerical Completeness (Zero NaN / Inf)",
            "violations": nan_x + nan_y + inf_x + inf_y,
            "status": "PASS" if num_clean else "FAIL",
        })

        # 5. Target Non-Negativity
        neg_targets = int((y < 0.0).sum())
        checks.append({
            "check": "Target Non-Negativity (PM2.5 >= 0)",
            "violations": neg_targets,
            "status": "PASS" if neg_targets == 0 else "FAIL",
        })

        all_passed = all(c["status"] == "PASS" for c in checks)
        summary = {
            "tensor_validation_status": "PASS" if all_passed else "FAIL",
            "total_sequences": len(X),
            "sequence_window": window_size,
            "feature_dimension": X.shape[2] if len(X.shape) == 3 else 0,
            "checks": checks,
        }

        return all_passed, summary

    def verify_architecture_smoke_pass(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """Simulates lightweight forward-pass tensor consumption for LSTM, TCN, and Transformer."""
        arch_results = {}
        batch_size = 32
        N, W, D = X.shape

        X_batch = X[:batch_size]
        y_batch = y[:batch_size]

        # 1. LSTM Temporal Recurrence Forward Pass Simulation
        # Hidden state recurrent projection: (B, W, D) -> (B, H) -> (B, 1)
        w_recurrent = np.ones((D, 64), dtype=np.float32) * 0.01
        h_lstm = np.tanh(np.mean(X_batch, axis=1) @ w_recurrent) # (B, 64)
        out_lstm = np.maximum(h_lstm @ np.ones((64, 1), dtype=np.float32) * 0.05, 0.0)
        lstm_valid = (out_lstm.shape == (batch_size, 1) and not np.isnan(out_lstm).any())
        arch_results["LSTM"] = "PASS" if lstm_valid else "FAIL"

        # 2. TCN Temporal Convolution Forward Pass Simulation
        # Multi-scale receptive fields: (B, W, D) -> (B, D_conv) -> (B, 1)
        conv1 = np.mean(X_batch[:, -3:, :], axis=1)
        conv2 = np.mean(X_batch[:, -7:, :], axis=1)
        conv_cat = np.hstack([conv1, conv2]) # (B, 2*D)
        w_tcn = np.ones((2 * D, 64), dtype=np.float32) * 0.01
        h_tcn = np.maximum(conv_cat @ w_tcn, 0.0)
        out_tcn = np.maximum(h_tcn @ np.ones((64, 1), dtype=np.float32) * 0.05, 0.0)
        tcn_valid = (out_tcn.shape == (batch_size, 1) and not np.isnan(out_tcn).any())
        arch_results["TCN"] = "PASS" if tcn_valid else "FAIL"

        # 3. Transformer Temporal Self-Attention Forward Pass Simulation
        # Scaled dot-product self-attention: (B, W, D) -> (B, W, D) -> (B, 1)
        scores = np.einsum("bwd,btd->bwt", X_batch, X_batch) / np.sqrt(D)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-8)
        attn_out = np.einsum("bwt,btd->bwd", attn, X_batch)
        h_trans = np.mean(attn_out, axis=1)
        out_trans = np.maximum(h_trans @ np.ones((D, 1), dtype=np.float32) * 0.05, 0.0)
        trans_valid = (out_trans.shape == (batch_size, 1) and not np.isnan(out_trans).any())
        arch_results["Transformer"] = "PASS" if trans_valid else "FAIL"

        all_passed = all(v == "PASS" for v in arch_results.values())
        return all_passed, arch_results
