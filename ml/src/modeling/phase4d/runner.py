import sys
import json
import hashlib
import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase4d.scenario_registry import ScenarioRegistryPhase4D
from ml.src.modeling.phase4d.feature_intervention import FeatureInterventionEnginePhase4D
from ml.src.modeling.phase4d.counterfactual_engine import CounterfactualSimulationEnginePhase4D
from ml.src.modeling.phase4d.plausibility import PlausibilityValidatorPhase4D
from ml.src.modeling.phase4d.ood_detector import OODDetectorPhase4D
from ml.src.modeling.phase4d.shap_consistency import SHAPConsistencyPhase4D
from ml.src.modeling.phase4d.interaction_analysis import InteractionAnalysisPhase4D
from ml.src.modeling.phase4d.confidence import ConfidenceEvaluatorPhase4D
from ml.src.modeling.phase4d.event_counterfactual import EventCounterfactualPhase4D
from ml.src.modeling.phase4d.visualization import VisualizationEnginePhase4D
from ml.src.modeling.phase4d.report_generator import ReportGeneratorPhase4D

logger = setup_logger("MasterRunnerPhase4D")


class MasterRunnerPhase4D:
    """
    AtmosIQ Phase 4D Master Orchestrator.
    Executes source-category counterfactual simulations, interaction analysis, OOD detection, plausibility checks, confidence scoring, plots, and report generation.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.v1_frozen = Path("ml/data/modeling/v1/feature_dataset_frozen.csv")
        self.v2_frozen = Path("ml/data/modeling/v2/feature_dataset_frozen.csv")
        self.model_path = Path("ml/models/attribution/v1/model.joblib")
        self.p4b_group_file = Path("ml/experiments/phase4b/group_attributions/group_attributions_all.csv")
        self.p4c_catalog_file = Path("ml/experiments/phase4c/event_catalog.csv")
        self.p4c_conf_file = Path("ml/experiments/phase4c/confidence_scores.csv")

        self.v1_expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
        self.v2_expected_hash = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"

    def calculate_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_inputs(self):
        """Validates input datasets, model, and upstream artifacts."""
        logger.info("Validating Phase 4D input dependencies...")
        assert self.v1_frozen.exists(), "Dataset v1 missing!"
        assert self.v2_frozen.exists(), "Dataset v2 missing!"
        assert self.model_path.exists(), "Frozen model missing!"
        assert self.p4b_group_file.exists(), "Phase 4B group attributions missing!"
        assert self.p4c_catalog_file.exists(), "Phase 4C event catalog missing!"

        v1_hash = self.calculate_sha256(self.v1_frozen)
        v2_hash = self.calculate_sha256(self.v2_frozen)
        assert v1_hash == self.v1_expected_hash, "Dataset v1 modified!"
        assert v2_hash == self.v2_expected_hash, "Dataset v2 modified!"

        logger.info("Phase 4D Input Validation 100% PASS.")

    def write_metadata(self, model_hash: str, scenarios_dict: dict, high_conf_pct: float):
        """Writes ml/experiments/phase4d/metadata.json."""
        metadata = {
            "phase": "Phase 4D",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_type": "random_forest",
            "model_hash": model_hash,
            "dataset_version": "v2",
            "dataset_hash": "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301",
            "feature_count": 147,
            "observations_simulated": 1827,
            "scenarios_evaluated": list(scenarios_dict.keys()),
            "high_confidence_counterfactual_pct": high_conf_pct,
            "ready_for_phase4e": True
        }
        with open(self.exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata exported to {self.exp_dir / 'metadata.json'}.")

    def run(self):
        """Executes full Phase 4D master pipeline."""
        logger.info("=== Starting AtmosIQ Phase 4D Master Pipeline ===")

        # 1. Validate inputs
        self.validate_inputs()
        model_hash = self.calculate_sha256(self.model_path)

        df = pd.read_csv(self.v2_frozen)
        group_shap_df = pd.read_csv(self.p4b_group_file)
        event_catalog_df = pd.read_csv(self.p4c_catalog_file)
        p4c_conf_df = pd.read_csv(self.p4c_conf_file)

        # 2. Export scenario registry
        registry = ScenarioRegistryPhase4D(exp_dir=str(self.exp_dir))
        registry.export_registry_json()
        scenarios_dict = registry.scenarios

        # 3. Initialize Feature Intervention Engine
        intervention_engine = FeatureInterventionEnginePhase4D()
        X_all = df[intervention_engine.feature_order]

        # 4. Master Counterfactual Simulations
        sim_engine = CounterfactualSimulationEnginePhase4D(exp_dir=str(self.exp_dir))
        cf_results_df, summary_df = sim_engine.run_simulations(intervention_engine, scenarios_dict, X_all, df)

        # 5. Plausibility Validation
        plaus_val = PlausibilityValidatorPhase4D(exp_dir=str(self.exp_dir))
        plaus_rows = []
        X_mat = X_all[intervention_engine.feature_order].values

        for i in range(len(cf_results_df)):
            cf_row = cf_results_df.iloc[i]
            dt = cf_row["date"]
            scen = cf_row["scenario"]
            grp = cf_row["target_group"]

            d_idx = df[df["date"] == dt].index[0]
            x_obs = X_mat[d_idx]
            if grp == "multi_group":
                x_cf = intervention_engine.apply_multi_group_intervention(x_obs, {"biomass_burning": "q25", "wind_ventilation": "q75"})
            else:
                x_cf = intervention_engine.apply_intervention(x_obs, grp, "q25")

            res_p = plaus_val.validate_counterfactual_plausibility(intervention_engine.feature_order, intervention_engine.group_mapping, x_obs, x_cf, grp, intervention_engine.reference_quantiles)
            res_p["date"] = dt
            res_p["scenario"] = scen
            plaus_rows.append(res_p)

        plausibility_df = pd.DataFrame(plaus_rows)
        plausibility_df.to_csv(self.exp_dir / "plausibility_checks.csv", index=False)

        # 6. OOD Detection
        ood_det = OODDetectorPhase4D(exp_dir=str(self.exp_dir))
        ood_det.fit_reference_distribution(X_all, intervention_engine.feature_order)
        ood_rows = []

        for i in range(len(cf_results_df)):
            cf_row = cf_results_df.iloc[i]
            dt = cf_row["date"]
            scen = cf_row["scenario"]
            grp = cf_row["target_group"]

            d_idx = df[df["date"] == dt].index[0]
            x_obs = X_mat[d_idx]
            if grp == "multi_group":
                x_cf = intervention_engine.apply_multi_group_intervention(x_obs, {"biomass_burning": "q25", "wind_ventilation": "q75"})
            else:
                x_cf = intervention_engine.apply_intervention(x_obs, grp, "q25")

            f_flag, f_score, f_reason = ood_det.evaluate_ood(x_cf)
            ood_rows.append({
                "date": dt,
                "scenario": scen,
                "ood_flag": f_flag,
                "ood_score": f_score,
                "ood_reason": f_reason
            })

        ood_df = pd.DataFrame(ood_rows)
        ood_df.to_csv(self.exp_dir / "ood_analysis.csv", index=False)

        # 7. SHAP Consistency Evaluation
        shap_eval = SHAPConsistencyPhase4D(exp_dir=str(self.exp_dir))
        consistency_df = shap_eval.evaluate_consistency(group_shap_df, cf_results_df)

        # 8. Multi-Group Interaction Analysis
        inter_eval = InteractionAnalysisPhase4D(exp_dir=str(self.exp_dir))
        inter_df = inter_eval.compute_interactions(intervention_engine, X_all, df["date"].tolist())

        # 9. Confidence Evaluation
        conf_eval = ConfidenceEvaluatorPhase4D(exp_dir=str(self.exp_dir))
        conf_df = conf_eval.evaluate_confidence(cf_results_df, plausibility_df, ood_df, consistency_df, p4c_conf_df)

        # 10. Event Counterfactual Analysis & API
        evt_cf = EventCounterfactualPhase4D(exp_dir=str(self.exp_dir))
        evt_cf_df = evt_cf.analyze_event_counterfactuals(event_catalog_df, intervention_engine, X_all, df)
        evt_cf.explain_counterfactual_date("2024-11-16", "biomass_low", df, X_all, intervention_engine, conf_df)

        # 11. Research Diagnostic Plots
        viz = VisualizationEnginePhase4D(exp_dir=str(self.exp_dir))
        viz.generate_all_plots(cf_results_df, summary_df, inter_df, evt_cf_df, ood_df, conf_df)

        # 12. Reports & Metadata
        report_gen = ReportGeneratorPhase4D(exp_dir=str(self.exp_dir))
        case_studies = report_gen.select_case_studies(cf_results_df, df, group_shap_df)

        high_conf_pct = float(np.mean(conf_df["counterfactual_confidence_level"] == "HIGH")) * 100
        self.write_metadata(model_hash, scenarios_dict, high_conf_pct)
        report_gen.generate_report(summary_df, inter_df, evt_cf_df, conf_df, case_studies, model_hash)

        logger.info("=== Phase 4D Master Pipeline Completed Successfully ===")


if __name__ == "__main__":
    runner = MasterRunnerPhase4D()
    runner.run()
