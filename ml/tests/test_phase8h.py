"""
Unit and Integration Tests for AtmosIQ Phase 8H (Final Deep-Learning Pipeline Validation Gate).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8h import (
    Phase8HConfig,
    Phase8HProvenanceManager,
    Phase8HLSTMModel,
    Phase8HTCNModel,
    Phase8HTransformerModel,
    Phase8HSequenceDataset,
    Phase8HDataLoader,
    Phase8HTrainer,
    Phase8HAuditor,
)
from ml.src.modeling.phase8g.sequence_builder import Phase8GSequenceBuilder
from ml.src.modeling.phase8g.policy_engine import Phase8GAugmentationPolicyEngine, AugmentationPolicyViolation


class TestPhase8H:
    @classmethod
    def setup_class(cls):
        cls.config = Phase8HConfig()
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

        cls.prov_mgr = Phase8HProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.policy_engine = Phase8GAugmentationPolicyEngine(0.25, 0.50)
        cls.seq_builder = Phase8GSequenceBuilder(cls.feature_registry, "pm25")
        cls.seq_builder.fit_scaler(cls.df_real_train)
        cls.auditor = Phase8HAuditor(cls.feature_registry)

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

        dataset = Phase8HSequenceDataset(X, y, df_prov.to_dict(orient="records"))
        assert len(dataset) == 896
        loader = Phase8HDataLoader(dataset, batch_size=32, shuffle=True, seed=42)
        assert len(loader) == int(np.ceil(896 / 32))

    # 4. Model Architectures Forward & Backward Passes
    @pytest.mark.parametrize("model_cls", [Phase8HLSTMModel, Phase8HTCNModel, Phase8HTransformerModel])
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
        model = Phase8HLSTMModel(window_size=14, feature_dim=35, seed=42)
        trainer = Phase8HTrainer(model, lr=0.001, seed=42)
        X_dummy = np.random.randn(8, 14, 35).astype(np.float32)
        preds_before = model.forward(X_dummy)

        ckpt_file = tmp_path / "test_ckpt.json"
        trainer.save_checkpoint(ckpt_file, corpus_sha="test_sha", contract_version="v1.1.0")

        # Reload into new instance
        reloaded = Phase8HLSTMModel(window_size=14, feature_dim=35, seed=42)
        meta = trainer.load_checkpoint(ckpt_file, reloaded)
        preds_after = reloaded.forward(X_dummy)

        assert meta["corpus_sha256"] == "test_sha"
        delta = float(np.max(np.abs(preds_before - preds_after)))
        assert delta <= 1e-9

    # 6. Data Isolation Audit
    def test_data_isolation(self):
        df_prov = pd.DataFrame({"source_partition": ["2020-2021"] * 10})
        passed, df_aud = self.auditor.audit_leakage(self.df_real_train, self.df_real_test, df_prov)
        assert passed is True
        assert (df_aud["violations"] == 0).all()

    # 7. System Resource Audit
    def test_resource_audit(self):
        passed, df_res = self.auditor.audit_resources()
        assert passed is True
        assert len(df_res) >= 4
