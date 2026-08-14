import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ShapEnginePhase4I")


class TreeShapEnginePhase4I:
    """
    TreeSHAP Computation and Reconstruction Validation Engine for Phase 4I.
    Trains / loads the exact Phase 4H Promoted Random Forest (35 features) and computes SHAP values.
    """

    PROMOTED_PARAMS = {
        "n_estimators": 400,
        "max_depth": 9,
        "min_samples_split": 4,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "random_state": 42
    }

    def __init__(self, df_v3: pd.DataFrame, features_35: list, group_df: pd.DataFrame):
        self.df_v3 = df_v3.copy()
        self.df_v3['date'] = pd.to_datetime(self.df_v3['date'])
        self.features = features_35
        self.group_df = group_df
        self.feature_to_group = dict(zip(group_df['feature'], group_df['group']))

        # Ensure train data split (2020-2023)
        train_mask = self.df_v3['date'].dt.year.isin([2020, 2021, 2022, 2023])
        self.X_train = self.df_v3.loc[train_mask, self.features].fillna(0.0)
        self.y_train = self.df_v3.loc[train_mask, 'pm25'].values

    def get_or_fit_model(self, model_save_path: Path) -> RandomForestRegressor:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        if model_save_path.exists():
            logger.info(f"Loading Phase 4H Promoted RF Model from {model_save_path}...")
            model = joblib.load(model_save_path)
        else:
            logger.info("Fitting exact Phase 4H Promoted RF Model on 2020-2023 Dataset v3...")
            model = RandomForestRegressor(**self.PROMOTED_PARAMS)
            model.fit(self.X_train, self.y_train)
            joblib.dump(model, model_save_path)
            logger.info(f"Saved Promoted Model to {model_save_path}")

        return model

    def run_shap_analysis(self, model: RandomForestRegressor, exp_dir: Path) -> dict:
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Computing TreeSHAP explanations for Dataset v3...")

        X_all = self.df_v3[self.features].fillna(0.0)
        preds = model.predict(X_all)

        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_all)
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = float(expected_value[0])

        # 1. SHAP Reconstruction Validation
        shap_sums = shap_values_raw.sum(axis=1)
        reconstructed_preds = expected_value + shap_sums
        reconstruction_errors = np.abs(preds - reconstructed_preds)

        max_err = float(np.max(reconstruction_errors))
        mean_err = float(np.mean(reconstruction_errors))
        median_err = float(np.median(reconstruction_errors))
        p95_err = float(np.percentile(reconstruction_errors, 95))

        logger.info(f"SHAP Reconstruction Errors: Max={max_err:.6e}, Mean={mean_err:.6e}, Median={median_err:.6e}, P95={p95_err:.6e}")
        assert max_err <= 1e-4, f"TreeSHAP Reconstruction Failed! Max error = {max_err:.6f} > 1e-4"

        df_reconstruction = pd.DataFrame([{
            "max_reconstruction_error": max_err,
            "mean_reconstruction_error": mean_err,
            "median_reconstruction_error": median_err,
            "p95_reconstruction_error": p95_err,
            "tolerance": 1e-4,
            "validation_status": "PASS" if max_err <= 1e-4 else "FAIL"
        }])
        df_reconstruction.to_csv(exp_dir / "v3_shap_reconstruction.csv", index=False)

        # 2. Build SHAP DataFrames
        dates = self.df_v3['date'].dt.strftime('%Y-%m-%d').values
        actuals = self.df_v3['pm25'].values
        years = self.df_v3['date'].dt.year.values

        df_shap_all = pd.DataFrame(shap_values_raw, columns=self.features)
        df_shap_all.insert(0, "date", dates)
        df_shap_all.insert(1, "year", years)
        df_shap_all.insert(2, "actual_pm25", actuals)
        df_shap_all.insert(3, "predicted_pm25", preds)
        df_shap_all.insert(4, "base_value", expected_value)

        df_shap_all.to_csv(exp_dir / "v3_shap_values_all.csv", index=False)

        val_mask = (years >= 2022) & (years <= 2023)
        test_mask = (years == 2024)

        df_shap_all[val_mask].to_csv(exp_dir / "v3_shap_values_validation.csv", index=False)
        df_shap_all[test_mask].to_csv(exp_dir / "v3_shap_values_test.csv", index=False)

        # 3. Global Feature Importance
        abs_shap = np.abs(shap_values_raw)
        mean_abs = abs_shap.mean(axis=0)
        mean_signed = shap_values_raw.mean(axis=0)
        std_shap = shap_values_raw.std(axis=0)

        pos_freq = (shap_values_raw > 0).mean(axis=0)
        neg_freq = (shap_values_raw < 0).mean(axis=0)

        feat_imp_df = pd.DataFrame({
            "feature": self.features,
            "group": [self.feature_to_group.get(f, "other") for f in self.features],
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "std_shap": std_shap,
            "positive_contrib_freq": pos_freq,
            "negative_contrib_freq": neg_freq
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        feat_imp_df['rank'] = np.arange(1, len(feat_imp_df) + 1)
        feat_imp_df.to_csv(exp_dir / "v3_global_feature_importance.csv", index=False)

        # 4. Group Level Aggregation
        unique_groups = sorted(list(set(self.feature_to_group.values())))
        group_shap_matrix = np.zeros((len(self.df_v3), len(unique_groups)))

        for i, grp in enumerate(unique_groups):
            grp_feats = [f for f in self.features if self.feature_to_group.get(f) == grp]
            if grp_feats:
                group_shap_matrix[:, i] = df_shap_all[grp_feats].sum(axis=1).values

        df_group_shap_all = pd.DataFrame(group_shap_matrix, columns=unique_groups)
        df_group_shap_all.insert(0, "date", dates)
        df_group_shap_all.insert(1, "year", years)
        df_group_shap_all.insert(2, "actual_pm25", actuals)
        df_group_shap_all.insert(3, "predicted_pm25", preds)

        df_group_shap_all.to_csv(exp_dir / "v3_group_attributions_all.csv", index=False)
        df_group_shap_all[test_mask].to_csv(exp_dir / "v3_group_attributions_test.csv", index=False)

        # Group Importance
        grp_mean_abs = np.abs(group_shap_matrix).mean(axis=0)
        grp_mean_signed = group_shap_matrix.mean(axis=0)
        grp_pos_freq = (group_shap_matrix > 0).mean(axis=0)

        extreme_mask = actuals >= 150.0
        grp_extreme_mean = group_shap_matrix[extreme_mask].mean(axis=0) if extreme_mask.sum() > 0 else np.zeros(len(unique_groups))

        grp_imp_df = pd.DataFrame({
            "attribution_group": unique_groups,
            "mean_abs_shap": grp_mean_abs,
            "mean_signed_shap": grp_mean_signed,
            "positive_contrib_freq": grp_pos_freq,
            "extreme_day_mean_shap": grp_extreme_mean
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        grp_imp_df['rank'] = np.arange(1, len(grp_imp_df) + 1)
        grp_imp_df.to_csv(exp_dir / "v3_group_importance.csv", index=False)

        logger.info("TreeSHAP analysis and group attribution completed cleanly.")
        return {
            "model": model,
            "explainer": explainer,
            "shap_values_raw": shap_values_raw,
            "expected_value": expected_value,
            "df_shap_all": df_shap_all,
            "df_group_shap_all": df_group_shap_all,
            "feat_imp_df": feat_imp_df,
            "grp_imp_df": grp_imp_df,
            "reconstruction_max_err": max_err
        }
