"""
Unit and Integration Tests for AtmosIQ Phase 9 (Deep Learning Training & Selection).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase9 import (
    Phase9Config,
    Phase9ProvenanceManager,
    Phase9LSTMModel,
    Phase9TCNModel,
    Phase9TransformerModel,
    Phase9SequenceDataset,
    Phase9DataLoader,
    Phase9Trainer,
    Phase9Evaluator,
    Phase9ModelSelector,
)
from ml.src.modeling.phase8g.sequence_builder import Phase8GSequenceBuilder
from ml.src.modeling.phase8g.policy_engine import Phase8GAugmentationPolicyEngine, AugmentationPolicyViolation


class TestPhase9:
    @classmethod
    def setup_class(cls):
        cls.config = Phase9Config()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()

        df_full = pd.read_csv(cls.config.dataset_v3_path)
        cls.df_real_train = df_full[
            (df_full["date"] >= cls.config.dev_train_start_date) &
            (df_full["date"] <= cls.config.dev_train_end_date)
        ].copy()
        cls.df_real_test = df_full[
            (df_full["date"] >= cls.config.locked_eval_start_date) &
            (df_full["date"] <= cls.config.locked_eval_end_date)
        ].copy()

        cls.df_8d_corpus = pd.read_parquet(cls.config.phase8d_corpus_path)

        cls.prov_mgr = Phase9ProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.policy_engine = Phase8GAugmentationPolicyEngine(0.25, 0.50)
        cls.seq_builder = Phase8GSequenceBuilder(cls.feature_registry, "pm25")
        cls.seq_builder.fit_scaler(cls.df_real_train)
        cls.evaluator = Phase9Evaluator(extreme_threshold=250.0)
        cls.selector = Phase9ModelSelector(cls.config.manifests_dir)

    # 1. Protected Artifact Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Augmentation Policy Engine Enforcement
    def test_augmentation_policy_enforcement(self):
        res_25 = self.policy_engine.validate_augmentation_request(0.25)
        assert res_25["tier"] == "RECOMMENDED_PRODUCTION"

        with pytest.raises(AugmentationPolicyViolation):
            self.policy_engine.validate_augmentation_request(1.00)

    # 3. Temporal Sequence Construction
    def test_sequence_construction_and_dataset(self):
        X, y, df_prov, meta = self.seq_builder.assemble_integrated_dataset(
            self.df_real_train, self.df_8d_corpus, augmentation_ratio=0.25, window_size=14, seed=42
        )
        assert len(X) == 896
        assert X.shape == (896, 14, 35)
        assert len(y) == 896

        dataset = Phase9SequenceDataset(X, y, df_prov.to_dict(orient="records"))
        assert len(dataset) == 896
        loader = Phase9DataLoader(dataset, batch_size=32, shuffle=True, seed=42)
        assert len(loader) == int(np.ceil(896 / 32))

    # 4. Model Architectures Forward & Backward Passes
    @pytest.mark.parametrize("model_cls", [Phase9LSTMModel, Phase9TCNModel, Phase9TransformerModel])
    def test_model_forward_backward_pass(self, model_cls):
        model = model_cls(window_size=14, feature_dim=35, seed=42)
        X_dummy = np.random.randn(8, 14, 35).astype(np.float32)
        y_dummy = (np.random.rand(8) * 50.0).astype(np.float32)

        # Forward
        out = model.forward(X_dummy)
        assert out.shape == (8,)
        assert not np.isnan(out).any()

        # Backward & Loss
        loss, grads = model.compute_loss_and_backward(X_dummy, y_dummy)
        assert loss >= 0.0
        assert not np.isnan(loss)
        assert len(grads) == len(model.params)

        # Gradient stats
        grad_stats = model.get_gradient_stats()
        assert grad_stats["has_nan"] is False
        assert grad_stats["has_inf"] is False

    # 5. Checkpoint Round-Trip Verification
    def test_checkpoint_round_trip(self, tmp_path):
        model = Phase9LSTMModel(window_size=14, feature_dim=35, seed=42)
        trainer = Phase9Trainer(model, lr=0.001, seed=42)
        X_dummy = np.random.randn(8, 14, 35).astype(np.float32)
        preds_before = model.forward(X_dummy)

        ckpt_file = tmp_path / "test_ckpt.json"
        trainer.save_checkpoint(ckpt_file, corpus_sha="test_sha", contract_version="v1.1.0")

        # Reload into new instance
        reloaded = Phase9LSTMModel(window_size=14, feature_dim=35, seed=42)
        meta = trainer.load_checkpoint(ckpt_file, reloaded)
        preds_after = reloaded.forward(X_dummy)

        assert meta["corpus_sha256"] == "test_sha"
        delta = float(np.max(np.abs(preds_before - preds_after)))
        assert delta <= 1e-9

    # 6. Evaluation Metrics & Extreme-Event Filtering
    def test_evaluation_metrics(self):
        y_true = np.array([50.0, 100.0, 260.0, 300.0], dtype=np.float32)
        y_pred = np.array([52.0, 95.0, 250.0, 290.0], dtype=np.float32)

        metrics = self.evaluator.evaluate_metrics(y_true, y_pred)
        assert metrics["mae"] > 0.0
        assert metrics["rmse"] > 0.0
        assert metrics["extreme_count"] == 2
        assert metrics["extreme_mae"] == 10.0

    # 7. Model Ranking & Selection Engine
    def test_model_ranking_and_selection(self, tmp_path):
        val_records = [
            {
                "exp_id": "M1",
                "architecture": "LSTM",
                "augmentation_ratio": 0.25,
                "corpus": "CAL-07",
                "seed": 42,
                "val_mae": 25.0,
                "val_rmse": 32.0,
                "val_r2": 0.65,
                "val_extreme_mae": 45.0,
            },
            {
                "exp_id": "M2",
                "architecture": "TCN",
                "augmentation_ratio": 0.25,
                "corpus": "CAL-07",
                "seed": 42,
                "val_mae": 30.0,
                "val_rmse": 38.0,
                "val_r2": 0.55,
                "val_extreme_mae": 55.0,
            },
        ]
        selector = Phase9ModelSelector(tmp_path)
        ranked_df = selector.rank_models(val_records)
        assert ranked_df.iloc[0]["exp_id"] == "M1"

        manifest = selector.select_winning_candidate(ranked_df, {"test_mae": 24.5})
        assert manifest["selected_architecture"] == "LSTM"
        assert (tmp_path / "phase9_model_selection.json").exists()
