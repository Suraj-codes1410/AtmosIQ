import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase6f.config import DecisionSupportConfigPhase6F
from ml.src.modeling.phase6f.attribution_adapter import AttributionAdapterPhase6F

logger = setup_logger("CounterfactualAdapterPhase6F")


class CounterfactualAdapterPhase6F:
    """
    Production Counterfactual Simulation Adapter for Phase 6F.
    Applies validated environmental scenarios and provides response predictions with uncertainty metadata.
    """

    def __init__(
        self,
        config: DecisionSupportConfigPhase6F,
        model_path: str = "ml/models/production/v3/model.joblib",
        dataset_path: str = "ml/data/modeling/v3/feature_dataset_frozen.csv",
        cf_cases_path: str = "ml/experiments/phase6e/counterfactual_cases.csv"
    ):
        self.config = config
        self.model = joblib.load(Path(model_path))
        self.df_train = pd.read_csv(Path(dataset_path))
        self.group_defs = AttributionAdapterPhase6F.GROUP_DEFINITIONS

        # Compute historical reference quantiles (2020-2021 historical baseline)
        train_baseline = self.df_train[pd.to_datetime(self.df_train['date']).dt.year.isin([2020, 2021])].copy()
        if len(train_baseline) == 0:
            train_baseline = self.df_train.copy()

        self.q25_dict = {col: float(train_baseline[col].quantile(0.25)) for col in train_baseline.columns if pd.api.types.is_numeric_dtype(train_baseline[col])}
        self.q50_dict = {col: float(train_baseline[col].quantile(0.50)) for col in train_baseline.columns if pd.api.types.is_numeric_dtype(train_baseline[col])}
        self.q75_dict = {col: float(train_baseline[col].quantile(0.75)) for col in train_baseline.columns if pd.api.types.is_numeric_dtype(train_baseline[col])}

        # Load Phase 6E scenario calibration metadata
        self.scenario_metadata = {}
        if Path(cf_cases_path).exists():
            df_cf = pd.read_csv(cf_cases_path)
            for _, r in df_cf.iterrows():
                self.scenario_metadata[r['scenario_name']] = {
                    "mean_delta_std": float(r['mean_delta_std']),
                    "mean_directional_stability": float(r['mean_directional_stability']),
                    "mean_interval_width_90": float(r['mean_interval_width_90'])
                }

    def simulate(self, x_vec: np.ndarray, feature_names: List[str], scenario_name: str = "combined_all_favorable") -> Dict[str, Any]:
        """
        Simulates counterfactual response for a specified scenario.
        """
        if x_vec.ndim == 1:
            X_orig = x_vec.reshape(1, -1)
        else:
            X_orig = x_vec.copy()

        X_cf = X_orig.copy()
        feat_to_idx = {name: i for i, name in enumerate(feature_names)}

        # Apply overrides
        intervened_groups = []
        if scenario_name == "biomass_low":
            intervened_groups = ["biomass_burning"]
            for f in self.group_defs["biomass_burning"]:
                if f in feat_to_idx and f in self.q25_dict:
                    X_cf[0, feat_to_idx[f]] = self.q25_dict[f]
        elif scenario_name == "biomass_median":
            intervened_groups = ["biomass_burning"]
            for f in self.group_defs["biomass_burning"]:
                if f in feat_to_idx and f in self.q50_dict:
                    X_cf[0, feat_to_idx[f]] = self.q50_dict[f]
        elif scenario_name == "biomass_high":
            intervened_groups = ["biomass_burning"]
            for f in self.group_defs["biomass_burning"]:
                if f in feat_to_idx and f in self.q75_dict:
                    X_cf[0, feat_to_idx[f]] = self.q75_dict[f]
        elif scenario_name == "wind_stagnant":
            intervened_groups = ["wind_ventilation"]
            for f in self.group_defs["wind_ventilation"]:
                if f in feat_to_idx and f in self.q25_dict:
                    X_cf[0, feat_to_idx[f]] = self.q25_dict[f]
        elif scenario_name == "wind_dispersion":
            intervened_groups = ["wind_ventilation"]
            for f in self.group_defs["wind_ventilation"]:
                if f in feat_to_idx and f in self.q75_dict:
                    X_cf[0, feat_to_idx[f]] = self.q75_dict[f]
        elif scenario_name == "meteorology_normal":
            intervened_groups = ["meteorology"]
            for f in self.group_defs["meteorology"]:
                if f in feat_to_idx and f in self.q50_dict:
                    X_cf[0, feat_to_idx[f]] = self.q50_dict[f]
        elif scenario_name == "combined_biomass_wind":
            intervened_groups = ["biomass_burning", "wind_ventilation"]
            for f in self.group_defs["biomass_burning"]:
                if f in feat_to_idx and f in self.q25_dict:
                    X_cf[0, feat_to_idx[f]] = self.q25_dict[f]
            for f in self.group_defs["wind_ventilation"]:
                if f in feat_to_idx and f in self.q75_dict:
                    X_cf[0, feat_to_idx[f]] = self.q75_dict[f]
        elif scenario_name == "combined_all_favorable":
            intervened_groups = ["biomass_burning", "wind_ventilation", "meteorology"]
            for f in self.group_defs["biomass_burning"]:
                if f in feat_to_idx and f in self.q25_dict:
                    X_cf[0, feat_to_idx[f]] = self.q25_dict[f]
            for f in self.group_defs["wind_ventilation"]:
                if f in feat_to_idx and f in self.q75_dict:
                    X_cf[0, feat_to_idx[f]] = self.q75_dict[f]
            for f in self.group_defs["meteorology"]:
                if f in feat_to_idx and f in self.q50_dict:
                    X_cf[0, feat_to_idx[f]] = self.q50_dict[f]
        else:
            raise ValueError(f"Unknown counterfactual scenario: {scenario_name}")

        pred_orig = float(max(0.0, self.model.predict(X_orig)[0]))
        pred_cf = float(max(0.0, self.model.predict(X_cf)[0]))
        delta_pm25 = float(pred_cf - pred_orig)

        meta = self.scenario_metadata.get(scenario_name, {
            "mean_delta_std": 2.0,
            "mean_directional_stability": 0.95,
            "mean_interval_width_90": 8.0
        })

        direction = "DECREASE" if delta_pm25 < -0.1 else ("INCREASE" if delta_pm25 > 0.1 else "NEUTRAL")
        cf_interval_80 = [delta_pm25 - 1.28 * meta["mean_delta_std"], delta_pm25 + 1.28 * meta["mean_delta_std"]]

        return {
            "scenario_name": scenario_name,
            "intervened_groups": intervened_groups,
            "baseline_prediction": pred_orig,
            "counterfactual_prediction": pred_cf,
            "delta_pm25": delta_pm25,
            "direction": direction,
            "counterfactual_std": meta["mean_delta_std"],
            "counterfactual_interval_80": [float(cf_interval_80[0]), float(cf_interval_80[1])],
            "directional_stability": meta["mean_directional_stability"],
            "unit": "µg/m³"
        }
