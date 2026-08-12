import sys
import json
import datetime
from pathlib import Path

# Ensure root workspace directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ml.src.utils.logger import setup_logger
from ml.src.modeling.feature_audit import FeatureAuditEngine

logger = setup_logger("FeatureSelectionPhase3C")


class FeatureSelectionEngine:
    """
    AtmosIQ Phase 3C: Feature Selection & Dimensionality Reduction Engine.
    Constructs controlled information group feature sets, correlation redundancy reduced sets,
    model-based top-N feature sets, and domain-aware process feature sets strictly using TRAIN data.
    """

    def __init__(
        self,
        modeling_dir: str = "ml/data/modeling/v1",
        exp_dir: str = "ml/experiments/phase3c"
    ):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.sel_dir = self.exp_dir / "selected_features"
        self.sel_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = "pm25"
        self.date_col = "date"

    def load_data(self) -> tuple[pd.DataFrame, list[str]]:
        """Loads train.csv and safe feature whitelist."""
        train_path = self.modeling_dir / "train.csv"
        assert train_path.exists(), f"Train dataset missing: {train_path}"
        df_train = pd.read_csv(train_path)

        _, safe_features = FeatureAuditEngine.load_safe_features()
        return df_train, safe_features

    def build_controlled_group_sets(self, safe_features: list[str]) -> dict[str, list[str]]:
        """Constructs controlled information group feature sets."""
        group_sets = {}

        # Set A: Persistence
        group_sets["set_a_persistence"] = ["pm25_lag_1d"]

        # Categorize safe features
        pm25_hist = []
        met_wind = []
        fire = []
        pollutants = []

        for col in safe_features:
            grp = FeatureAuditEngine.classify_feature_group(col)
            if grp == "pm25_historical":
                pm25_hist.append(col)
            elif grp in ["meteorological", "wind_ventilation"]:
                met_wind.append(col)
            elif grp == "fire_biomass_burning":
                fire.append(col)
            elif grp == "other_pollutant_historical":
                pollutants.append(col)

        # Set B: PM2.5 History
        group_sets["set_b_pm25_history"] = sorted(list(set(pm25_hist)))

        # Set C: PM2.5 + Meteorology & Wind
        group_sets["set_c_pm25_meteorology"] = sorted(list(set(pm25_hist + met_wind)))

        # Set D: PM2.5 + Meteorology + Fire
        group_sets["set_d_pm25_met_fire"] = sorted(list(set(pm25_hist + met_wind + fire)))

        # Set E: PM2.5 + Meteorology + Fire + Other Pollutants
        group_sets["set_e_pm25_met_fire_pollutants"] = sorted(list(set(pm25_hist + met_wind + fire + pollutants)))

        # Set F: Full Safe Feature Set (201 features)
        group_sets["set_f_full_safe"] = sorted(safe_features)

        return group_sets

    def build_redundancy_reduced_set(self, df_train: pd.DataFrame, safe_features: list[str], threshold: float = 0.95) -> list[str]:
        """Strategy 1: Removes pairwise correlated features (|r| >= 0.95) using TRAIN set correlation matrix."""
        logger.info(f"Building Redundancy-Reduced feature set (|r| < {threshold}) on train.csv...")

        X_tr = df_train[safe_features]
        corr_matrix = X_tr.corr(method="pearson").abs()

        cols = list(safe_features)
        drop_cols = set()

        for i in range(len(cols)):
            col_a = cols[i]
            if col_a in drop_cols:
                continue

            for j in range(i + 1, len(cols)):
                col_b = cols[j]
                if col_b in drop_cols:
                    continue

                if corr_matrix.loc[col_a, col_b] >= threshold:
                    # Keep col_a, drop col_b
                    drop_cols.add(col_b)

        reduced_set = [c for c in cols if c not in drop_cols]
        logger.info(f"Redundancy Reduction complete: Retained {len(reduced_set)} features out of {len(cols)} (Dropped {len(drop_cols)}).")
        return sorted(reduced_set)

    def build_model_based_top_sets(self, df_train: pd.DataFrame, safe_features: list[str]) -> dict[str, list[str]]:
        """Strategy 2: Ranks features on TRAIN set using Random Forest and extracts Top 20, 40, 60, 100 subsets."""
        logger.info("Building Model-Based Top-N feature sets using Random Forest ranking on train.csv...")

        X_tr = df_train[safe_features]
        y_tr = df_train[self.target_col]

        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)

        rf_imp = pd.DataFrame({
            "feature": safe_features,
            "importance": rf.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        top_sets = {}
        for n in [20, 40, 60, 100]:
            top_sets[f"top{n}"] = rf_imp.head(n)["feature"].tolist()

        return top_sets

    def build_domain_aware_reduced_set(self, safe_features: list[str]) -> list[str]:
        """Strategy 3: Construct manually justified interpretable domain-aware process feature set."""
        logger.info("Building Domain-Aware process reduced feature set...")

        # Representative domain features across 6 process modules:
        domain_candidates = [
            # 1. PM2.5 History
            "pm25_lag_1d", "pm25_lag_2d", "pm25_roll_mean_3d", "pm25_roll_mean_7d", "pm25_roll_max_14d",
            # 2. Ventilation & Wind Vector
            "wind_speed_kmh_roll_mean_30d", "wind_x", "wind_y", "wind_speed_change", "pressure_x_wind",
            # 3. Meteorology & Wash-out
            "temperature_c_roll_mean_3d", "humidity_pct_roll_mean_3d", "pressure_change", "is_raining", "consecutive_rain_days",
            # 4. Biomass Burning & Transport
            "fire_hotspot_count_lag_1d", "fire_hotspot_sum_7d", "wind_weighted_hotspot_transport_score", "fire_acceleration",
            # 5. Chemical Ratios
            "pm25_pm10_ratio", "no2_so2_ratio", "co_normalized",
            # 6. Seasonality & Calendar
            "is_stubble_season", "is_winter", "is_monsoon", "is_weekend", "days_until_diwali", "traffic_activity_proxy"
        ]

        # Filter to ensure all are in safe_features whitelist
        domain_set = [f for f in domain_candidates if f in safe_features]
        logger.info(f"Domain-Aware reduced feature set constructed with {len(domain_set)} features.")
        return sorted(domain_set)

    def save_feature_txt_files(self, all_sets: dict[str, list[str]]):
        """Saves selected feature lists to .txt files in selected_features/."""
        logger.info("Exporting feature list .txt files...")

        filename_map = {
            "set_b_pm25_history": "pm25_history.txt",
            "set_c_pm25_meteorology": "pm25_meteorology.txt",
            "set_d_pm25_met_fire": "pm25_meteorology_fire.txt",
            "domain_reduced": "domain_reduced.txt",
            "redundancy_reduced": "redundancy_reduced.txt",
            "top20": "top20.txt",
            "top40": "top40.txt",
            "top60": "top60.txt",
            "top100": "top100.txt",
            "set_f_full_safe": "full_safe.txt"
        }

        for set_key, feature_list in all_sets.items():
            fname = filename_map.get(set_key, f"{set_key}.txt")
            txt_path = self.sel_dir / fname
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(feature_list) + "\n")

    def run(self) -> dict[str, list[str]]:
        """Executes full Phase 3C feature selection engine."""
        logger.info("=== Starting Phase 3C Feature Selection Engine ===")

        df_train, safe_features = self.load_data()

        # 1. Controlled Information Group Sets
        group_sets = self.build_controlled_group_sets(safe_features)

        # 2. Strategy 1: Redundancy Reduced Set
        red_reduced = self.build_redundancy_reduced_set(df_train, safe_features, threshold=0.95)

        # 3. Strategy 2: Model-Based Top-N Sets
        top_sets = self.build_model_based_top_sets(df_train, safe_features)

        # 4. Strategy 3: Domain-Aware Set
        domain_set = self.build_domain_aware_reduced_set(safe_features)

        # Combine all feature sets
        all_feature_sets = {}
        all_feature_sets.update(group_sets)
        all_feature_sets["redundancy_reduced"] = red_reduced
        all_feature_sets.update(top_sets)
        all_feature_sets["domain_reduced"] = domain_set

        # Save feature txt files
        self.save_feature_txt_files(all_feature_sets)

        # Export Feature Set Registry JSON & Results CSV
        registry_data = {
            set_name: {
                "feature_count": len(f_list),
                "features": f_list
            }
            for set_name, f_list in all_feature_sets.items()
        }

        with open(self.exp_dir / "feature_set_registry.json", "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=4)

        results_rows = [
            {"feature_set": s_name, "feature_count": len(f_list)}
            for s_name, f_list in all_feature_sets.items()
        ]
        pd.DataFrame(results_rows).to_csv(self.exp_dir / "feature_selection_results.csv", index=False)

        logger.info(f"Feature set registry created with {len(all_feature_sets)} distinct feature sets.")
        logger.info("=== Phase 3C Feature Selection Engine Completed Successfully ===")
        return all_feature_sets


if __name__ == "__main__":
    engine = FeatureSelectionEngine()
    engine.run()
