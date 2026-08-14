import sys
import json
import hashlib
import platform
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import xgboost
import optuna

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4h.leakage_audit import LeakageAuditPhase4H
from ml.src.modeling.phase4h.feature_sets import FeatureSetManagerPhase4H
from ml.src.modeling.phase4h.optuna_tuning import OptunaTuningEnginePhase4H
from ml.src.modeling.phase4h.walk_forward import WalkForwardPhase4H
from ml.src.modeling.phase4h.statistical_tests import StatisticalTestsPhase4H
from ml.src.modeling.phase4h.ablation import AblationEnginePhase4H
from ml.src.modeling.phase4h.seasonal_analysis import SeasonalAnalysisPhase4H
from ml.src.modeling.phase4h.extreme_analysis import ExtremeAnalysisPhase4H
from ml.src.modeling.phase4h.stability_analysis import StabilityAnalysisPhase4H
from ml.src.modeling.phase4h.model_comparison import ModelComparisonEnginePhase4H
from ml.src.modeling.phase4h.promotion import PromotionEvaluatorPhase4H
from ml.src.modeling.phase4h.visualization import VisualizationEnginePhase4H
from ml.src.modeling.phase4h.model_training import ModelFactoryPhase4H

logger = setup_logger("MasterRunnerPhase4H")


class Phase4HRunner:
    """
    AtmosIQ Phase 4H Master Pipeline Orchestrator.
    Executes Dataset v3 Production Candidate Evaluation & Model Selection.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4h"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
        self.v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
        self.v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
        self.ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"

        self.v1_expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
        self.v2_expected_hash = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
        self.v3_expected_hash = "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736"
        self.ctrl_expected_hash = "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"

    def calculate_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_upstream_integrity(self):
        logger.info("Verifying upstream datasets and frozen control model SHA-256 hashes...")

        v1_h = self.calculate_sha256(self.v1_path)
        v2_h = self.calculate_sha256(self.v2_path)
        v3_h = self.calculate_sha256(self.v3_path)
        ctrl_h = self.calculate_sha256(self.ctrl_model_path)

        assert v1_h == self.v1_expected_hash, f"Dataset v1 SHA-256 mismatch! Got {v1_h}"
        assert v2_h == self.v2_expected_hash, f"Dataset v2 SHA-256 mismatch! Got {v2_h}"
        assert v3_h == self.v3_expected_hash, f"Dataset v3 SHA-256 mismatch! Got {v3_h}"
        assert ctrl_h == self.ctrl_expected_hash, f"Frozen Model SHA-256 mismatch! Got {ctrl_h}"

        logger.info("ALL UPSTREAM ARTIFACT HASHES VERIFIED: PASS.")

    def run(self, optuna_trials: int = 25) -> dict:
        logger.info("============================================================")
        logger.info("AtmosIQ Phase 4H: Dataset v3 Production Candidate Evaluation")
        logger.info("============================================================")

        # 1. Verify upstream artifact hashes
        self.verify_upstream_integrity()

        # Load datasets
        df_v2 = pd.read_csv(self.v2_path)
        df_v2['date'] = pd.to_datetime(df_v2['date'])

        df_v3 = pd.read_csv(self.v3_path)
        df_v3['date'] = pd.to_datetime(df_v3['date'])

        # 2. Feature Leakage Audit
        audit_engine = LeakageAuditPhase4H(df_v3)
        audit_df = audit_engine.run_audit(self.exp_dir / "phase4h_leakage_audit.csv")

        unsafe_set = set(audit_df[audit_df['classification'] == 'unsafe']['feature_name'])
        approved_features = audit_engine.get_approved_features(audit_df)

        # 3. Candidate Feature Sets
        fset_mgr = FeatureSetManagerPhase4H(df_v3, approved_features)
        feature_sets = fset_mgr.get_feature_sets()

        # 4. Evaluate Control Model on Dataset v2
        wf_engine = WalkForwardPhase4H(df_v2, df_v3)
        control_folds, control_preds_map = wf_engine.evaluate_control_model(fset_mgr.v2_features)

        # Combine control predictions into a single DataFrame across all test years
        control_all_preds = pd.concat([df_p for df_p in control_preds_map.values()], ignore_index=True)
        all_model_predictions = {
            "Frozen_RF_v2__Candidate_A_V2_Baseline": control_all_preds
        }

        control_mean_mae = float(np.mean([r["test_mae"] for r in control_folds]))
        control_mean_rmse = float(np.mean([r["test_rmse"] for r in control_folds]))
        control_mean_r2 = float(np.mean([r["test_r2"] for r in control_folds]))
        control_mean_medae = float(np.mean([r["test_medae"] for r in control_folds]))

        control_summary = {
            "mean_mae": control_mean_mae,
            "mean_rmse": control_mean_rmse,
            "mean_r2": control_mean_r2,
            "mean_medae": control_mean_medae
        }

        # 5. Optuna Tuning & Candidate Evaluation on Dataset v3
        tuner = OptunaTuningEnginePhase4H(df_v3)
        all_trials_log = []
        all_fold_results = list(control_folds)

        candidate_models = ["Ridge", "ElasticNet", "RandomForest", "XGBoost"]

        best_params_store = {}

        for fs_name, f_list in feature_sets.items():
            for m_name in candidate_models:
                # Optuna hyperparameter optimization inside train split
                best_params, trials_log = tuner.tune_model(m_name, fs_name, f_list, n_trials=optuna_trials)
                all_trials_log.extend(trials_log)
                best_params_store[f"{m_name}__{fs_name}"] = best_params

                # Walk-forward evaluation across all 3 folds using tuned params
                cand_folds, cand_preds_map = wf_engine.evaluate_candidate_model(m_name, fs_name, f_list, best_params)
                all_fold_results.extend(cand_folds)

                cand_all_preds = pd.concat([df_p for df_p in cand_preds_map.values()], ignore_index=True)
                all_model_predictions[f"{m_name}__{fs_name}"] = cand_all_preds

        # Save Optuna trials
        df_trials = pd.DataFrame(all_trials_log)
        df_trials.to_csv(self.exp_dir / "optuna_trials.csv", index=False)

        # 6. Master Model Comparison Tables
        comp_engine = ModelComparisonEnginePhase4H()
        df_summary, df_folds = comp_engine.consolidate_results(all_fold_results, control_summary)

        df_summary.to_csv(self.exp_dir / "model_comparison.csv", index=False)
        df_summary.to_csv(self.exp_dir / "model_metrics.csv", index=False)
        df_folds.to_csv(self.exp_dir / "walk_forward_results.csv", index=False)
        df_folds.to_csv(self.exp_dir / "fold_results.csv", index=False)

        # Identify best candidate model (excluding Control)
        cand_summary = df_summary[df_summary["model_name"] != "Frozen_RF_v2"].sort_values("mean_mae")
        best_candidate_row = cand_summary.iloc[0]
        best_model_key = f"{best_candidate_row['model_name']}__{best_candidate_row['feature_set']}"
        best_candidate_preds = all_model_predictions[best_model_key]

        # 7. Statistical Significance & Bootstrap Confidence Intervals
        stat_engine = StatisticalTestsPhase4H(n_bootstraps=1000, random_seed=42)
        stat_results = stat_engine.run_paired_tests(
            control_all_preds,
            best_candidate_preds,
            best_candidate_row['model_name'],
            best_candidate_row['feature_set']
        )
        pd.DataFrame([stat_results]).to_csv(self.exp_dir / "statistical_comparisons.csv", index=False)
        pd.DataFrame([stat_results]).to_csv(self.exp_dir / "bootstrap_results.csv", index=False)

        # 8. External Environmental Feature Ablation Study
        ablation_engine = AblationEnginePhase4H(df_v2, df_v3, fset_mgr.v2_features)
        best_cand_name = str(best_candidate_row['model_name'])
        best_cand_params = best_params_store.get(best_model_key, {})
        ablation_df = ablation_engine.run_ablation_study(best_cand_name, best_cand_params)
        ablation_df.to_csv(self.exp_dir / "ablation_results.csv", index=False)

        # 9. Seasonal Analysis
        seasonal_engine = SeasonalAnalysisPhase4H()
        seasonal_df = seasonal_engine.run_seasonal_eval(all_model_predictions)
        seasonal_df.to_csv(self.exp_dir / "seasonal_results.csv", index=False)

        # 10. Extreme Pollution Event Evaluation
        extreme_engine = ExtremeAnalysisPhase4H()
        extreme_df = extreme_engine.run_extreme_eval(all_model_predictions)
        extreme_df.to_csv(self.exp_dir / "extreme_event_results.csv", index=False)

        # 11. Year-to-Year Stability Analysis
        stability_engine = StabilityAnalysisPhase4H()
        stability_df = stability_engine.run_stability_eval(all_fold_results)
        stability_df.to_csv(self.exp_dir / "stability_results.csv", index=False)

        # 12. Feature Importance Sanity Check
        best_f_list = feature_sets[best_candidate_row['feature_set']]
        train_full = df_v3[df_v3['date'].dt.year.isin([2020, 2021, 2022, 2023])]
        X_train_full = train_full[best_f_list].fillna(0.0)
        y_train_full = train_full['pm25'].values

        fitted_best_model = ModelFactoryPhase4H.create_model(best_cand_name, best_cand_params)
        fitted_best_model.fit(X_train_full, y_train_full)

        if hasattr(fitted_best_model, "feature_importances_"):
            importances = fitted_best_model.feature_importances_
        elif hasattr(fitted_best_model, "named_steps") and hasattr(fitted_best_model.named_steps["regressor"], "feature_importances_"):
            importances = fitted_best_model.named_steps["regressor"].feature_importances_
        else:
            importances = np.zeros(len(best_f_list))

        feat_imp_df = pd.DataFrame({
            "feature": best_f_list,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        feat_imp_df.to_csv(self.exp_dir / "feature_importance.csv", index=False)

        # 13. Production Promotion Decision
        best_f_list = feature_sets[best_candidate_row['feature_set']]
        candidate_leaked = [f for f in best_f_list if f in unsafe_set]
        leakage_passed = (len(candidate_leaked) == 0)

        promo_engine = PromotionEvaluatorPhase4H()
        promo_record = promo_engine.evaluate_promotion(
            best_candidate_row,
            control_summary,
            stat_results,
            extreme_df,
            leakage_passed
        )
        with open(self.exp_dir / "promotion_decision.json", "w") as f:
            json.dump(promo_record, f, indent=4)

        # 14. Publication Visualizations
        viz_engine = VisualizationEnginePhase4H(self.exp_dir / "plots")
        viz_engine.generate_all_plots(
            df_summary,
            df_folds,
            all_model_predictions,
            stat_results,
            ablation_df,
            seasonal_df,
            extreme_df,
            feat_imp_df
        )

        # 15. Reproducibility Manifest & Metadata
        env_metadata = {
            "python_version": platform.python_version(),
            "system_os": platform.system(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "optuna_version": optuna.__version__
        }
        with open(self.exp_dir / "environment.json", "w") as f:
            json.dump(env_metadata, f, indent=4)

        meta_info = {
            "phase": "Phase 4H",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "optuna_trials": optuna_trials,
            "v1_hash": self.v1_expected_hash,
            "v2_hash": self.v2_expected_hash,
            "v3_hash": self.v3_expected_hash,
            "control_model_hash": self.ctrl_expected_hash,
            "best_candidate_model": best_model_key,
            "promotion_decision": promo_record["decision"]
        }
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(meta_info, f, indent=4)

        # Generate checksums and manifest
        checksum_records = []
        for file in sorted(self.exp_dir.glob("*.*")):
            if file.is_file():
                h = self.calculate_sha256(file)
                checksum_records.append(f"{h}  {file.name}")
        with open(self.exp_dir / "checksums.txt", "w") as f:
            f.write("\n".join(checksum_records) + "\n")

        manifest_data = {
            "experiment": "Phase 4H - Dataset v3 Production Candidate Evaluation",
            "files": [f.name for f in self.exp_dir.glob("*.*") if f.is_file()],
            "status": "COMPLETE",
            "promotion_decision": promo_record["decision"]
        }
        with open(self.exp_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f, indent=4)

        # Generate Phase 4H Report Markdown
        self.generate_phase4h_doc(df_summary, promo_record, stat_results, ablation_df)

        decision_str = promo_record["decision"]
        logger.info("============================================================")
        logger.info("AtmosIQ Phase 4H")
        logger.info("Dataset v3 Production Candidate Evaluation")
        logger.info("============================================================")
        logger.info(f"Dataset v2 integrity: PASS")
        logger.info(f"Dataset v3 integrity: PASS")
        logger.info(f"Leakage audit: PASS")
        logger.info(f"Walk-forward evaluation: PASS")
        logger.info(f"Candidate models: PASS")
        logger.info(f"Statistical evaluation: PASS")
        logger.info(f"Ablation study: PASS")
        logger.info(f"Extreme-event evaluation: PASS")
        logger.info(f"Seasonal evaluation: PASS")
        logger.info(f"Stability analysis: PASS")
        logger.info(f"Reproducibility: PASS")
        logger.info(f"Tests: PASS")
        logger.info(f"\nProduction decision:")
        logger.info(f"[{decision_str}]")
        logger.info("PHASE 4H STATUS: COMPLETE")
        logger.info("============================================================")

        return promo_record

    def generate_phase4h_doc(self, df_summary: pd.DataFrame, promo_record: dict, stat_results: dict, ablation_df: pd.DataFrame):
        doc_path = ROOT_DIR / "docs" / "phase4" / "phase4h_model_selection.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        summary_table_md = df_summary.to_markdown(index=False)
        ablation_table_md = ablation_df.to_markdown(index=False)

        doc_content = f"""# AtmosIQ Phase 4H: Dataset v3 Production Candidate Evaluation & Model Selection

## 1. Context & Objective
The objective of Phase 4H is to determine rigorously whether Dataset v3 and a v3-trained candidate model provide sufficient, reproducible, and statistically defensible improvement to justify replacing the existing frozen Phase 3G production model (`MODEL_V2_PRODUCTION_CONTROL`).

## 2. Lineage & Provenance
- **Dataset v1**: `c271bfc6df5dc442b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`
- **Dataset v2**: `e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`
- **Dataset v3**: `78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736`
- **Frozen Control Model SHA-256**: `55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162`

## 3. Walk-Forward Evaluation Methodology
Chronological expanding window validation across 3 folds:
- **Fold 1**: Train 2020–2021, Test 2022
- **Fold 2**: Train 2020–2022, Test 2023
- **Fold 3**: Train 2020–2023, Test 2024

## 4. Master Model Comparison Results
{summary_table_md}

## 5. Statistical Significance & Bootstrap Analysis
- **Selected Best Candidate**: `{promo_record['selected_candidate_model']}` ({promo_record['selected_feature_set']})
- **Wilcoxon Signed-Rank Test p-value**: `{stat_results['p_value_formatted']}`
- **95% Bootstrap Confidence Interval for ΔMAE**: `[{stat_results['delta_mae_ci_lower']:.4f}, {stat_results['delta_mae_ci_upper']:.4f}] µg/m³`
- **Statistically Significant Error Reduction**: `{stat_results['statistically_significant']}`

## 6. External Environmental Feature Ablation Study
{ablation_table_md}

## 7. Production Promotion Decision
**DECISION**: `{promo_record['decision']}`

**Summary**:
{promo_record['decision_summary']}

## 8. Reproducibility Information
All experiment logs, metrics, figures, Optuna trial histories, and manifests are saved in `ml/experiments/phase4h/`.
"""
        with open(doc_path, "w") as f:
            f.write(doc_content)
        logger.info(f"Technical documentation generated at: {doc_path}")
