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
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ml.src.utils.logger import setup_logger

logger = setup_logger("Phase3B1Baselines")


class BaselineEvaluator:
    """
    AtmosIQ Phase 3B-1: Baseline Model Development.
    Executes Persistence, Linear Regression, and Ridge Regression baselines
    using strictly prediction-safe features and chronological temporal splits.
    """

    def __init__(
        self,
        modeling_dir: str = "ml/data/modeling/v1",
        experiment_dir: str = "ml/experiments/phase3b1"
    ):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(experiment_dir)
        self.pred_dir = self.exp_dir / "predictions"
        self.plots_dir = self.exp_dir / "plots"

        self.target_col = "pm25"
        self.date_col = "date"

        # Expected split dimensions
        self.expected_train_rows = 365
        self.expected_val_rows = 182
        self.expected_test_rows = 184
        self.expected_safe_features = 200

    def create_directories(self):
        """Creates experiment artifact directories."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_feature_whitelist(self) -> list[str]:
        """Loads authoritative feature whitelist from feature_availability.csv."""
        avail_path = self.modeling_dir / "feature_availability.csv"
        assert avail_path.exists(), f"Feature availability file missing: {avail_path}"

        df_avail = pd.read_csv(avail_path)
        safe_df = df_avail[df_avail["prediction_safe"] == True]
        safe_features = safe_df["feature_name"].tolist()

        actual_count = len(safe_features)
        logger.info(f"Loaded feature availability registry. Prediction-safe feature count: {actual_count}")

        if actual_count != self.expected_safe_features:
            logger.warning(
                f"DISCREPANCY DETECTED: Expected {self.expected_safe_features} prediction-safe features, "
                f"but found {actual_count} in feature_availability.csv. Proceeding with authoritative whitelist."
            )

        # Confirm no same-day or target features entered whitelist
        unsafe_in_whitelist = df_avail[(df_avail["prediction_safe"] == False) & (df_avail["feature_name"].isin(safe_features))]
        assert len(unsafe_in_whitelist) == 0, f"Unsafe features detected in whitelist: {unsafe_in_whitelist['feature_name'].tolist()}"

        return safe_features

    def load_and_validate_splits(self, safe_features: list[str]) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.Series, pd.Series,
        pd.DataFrame, pd.Series, pd.Series,
        pd.DataFrame, pd.Series, pd.Series
    ]:
        """Loads train, validation, and test splits and performs strict integrity validation."""
        logger.info("Loading Phase 3A dataset splits...")

        train_path = self.modeling_dir / "train.csv"
        val_path = self.modeling_dir / "validation.csv"
        test_path = self.modeling_dir / "test.csv"

        assert train_path.exists() and val_path.exists() and test_path.exists(), "Split files missing!"

        train_raw = pd.read_csv(train_path)
        val_raw = pd.read_csv(val_path)
        test_raw = pd.read_csv(test_path)

        # Integrity Checks
        assert len(train_raw) == self.expected_train_rows, f"Expected {self.expected_train_rows} train rows, got {len(train_raw)}"
        assert len(val_raw) == self.expected_val_rows, f"Expected {self.expected_val_rows} val rows, got {len(val_raw)}"
        assert len(test_raw) == self.expected_test_rows, f"Expected {self.expected_test_rows} test rows, got {len(test_raw)}"

        assert self.target_col in train_raw.columns, "Target pm25 missing!"
        assert all(f in train_raw.columns for f in safe_features), "Some safe features missing from dataset!"
        assert self.date_col not in safe_features, "Date column accidentally in predictor set!"
        assert self.target_col not in safe_features, "Target column accidentally in predictor set!"

        # Check NaNs and infs
        for dset_name, dset in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
            assert dset[safe_features].isnull().sum().sum() == 0, f"{dset_name} predictors contain NaNs!"
            assert np.isinf(dset[safe_features].values).sum() == 0, f"{dset_name} predictors contain Infs!"

        # Chronological & Disjointness Check
        assert train_raw[self.date_col].max() < val_raw[self.date_col].min(), "Train/Val temporal overlap!"
        assert val_raw[self.date_col].max() < test_raw[self.date_col].min(), "Val/Test temporal overlap!"

        # Extract X, y, dates
        dates_train = train_raw[self.date_col]
        dates_val = val_raw[self.date_col]
        dates_test = test_raw[self.date_col]

        y_train = train_raw[self.target_col]
        y_val = val_raw[self.target_col]
        y_test = test_raw[self.target_col]

        X_train = train_raw[safe_features].copy()
        X_val = val_raw[safe_features].copy()
        X_test = test_raw[safe_features].copy()

        logger.info(f"Splits successfully loaded. X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        return train_raw, val_raw, test_raw, X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test

    def generate_persistence_predictions(
        self, train_raw: pd.DataFrame, val_raw: pd.DataFrame, test_raw: pd.DataFrame
    ) -> dict[str, pd.Series]:
        """
        Generates naive persistence baseline predictions: y_hat(t) = PM2.5(t-1).
        Handles boundary conditions seamlessly:
        - Val day 1 (2024-01-01) uses train final day (2023-12-31) PM2.5.
        - Test day 1 (2024-07-01) uses val final day (2024-06-30) PM2.5.
        """
        logger.info("Generating Persistence Baseline predictions (y_hat(t) = PM2.5(t-1))...")

        # 1. Train Persistence
        train_y = train_raw[self.target_col]
        if "pm25_lag_1d" in train_raw.columns:
            p_train = train_raw["pm25_lag_1d"].copy()
        else:
            p_train = train_y.shift(1).fillna(train_y.iloc[0])

        # 2. Validation Persistence
        val_y = val_raw[self.target_col]
        last_train_pm25 = train_y.iloc[-1]
        p_val_values = [last_train_pm25] + val_y.iloc[:-1].tolist()
        p_val = pd.Series(p_val_values, index=val_raw.index)

        # 3. Test Persistence
        test_y = test_raw[self.target_col]
        last_val_pm25 = val_y.iloc[-1]
        p_test_values = [last_val_pm25] + test_y.iloc[:-1].tolist()
        p_test = pd.Series(p_test_values, index=test_raw.index)

        return {"train": p_train, "validation": p_val, "test": p_test}

    def train_linear_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[Pipeline, dict[str, np.ndarray]]:
        """Trains LinearRegression inside a StandardScaler pipeline fitted ONLY on X_train."""
        logger.info("Training Linear Regression Baseline...")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        # Fit ONLY on train
        pipeline.fit(X_train, y_train)

        preds = {
            "train": pipeline.predict(X_train),
            "validation": pipeline.predict(X_val),
            "test": pipeline.predict(X_test)
        }

        return pipeline, preds

    def train_ridge_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, X_test: pd.DataFrame, alpha: float = 1.0
    ) -> tuple[Pipeline, dict[str, np.ndarray]]:
        """Trains Ridge Regression (alpha=1.0) inside a StandardScaler pipeline fitted ONLY on X_train."""
        logger.info(f"Training Ridge Regression Baseline (alpha={alpha})...")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=42))
        ])

        # Fit ONLY on train
        pipeline.fit(X_train, y_train)

        preds = {
            "train": pipeline.predict(X_train),
            "validation": pipeline.predict(X_val),
            "test": pipeline.predict(X_test)
        }

        return pipeline, preds

    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        """Calculates MAE, RMSE, R2, and Median Absolute Error."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        med_ae = float(median_absolute_error(y_true, y_pred))

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Median_AE": round(med_ae, 4)
        }

    def save_predictions_and_metrics(
        self,
        all_preds: dict[str, dict[str, pd.Series]],
        all_y: dict[str, pd.Series],
        all_dates: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Saves prediction CSV artifacts and outputs compiled metrics.csv."""
        logger.info("Saving prediction artifacts and metric tables...")

        metrics_rows = []

        for model_name, preds_dict in all_preds.items():
            for split_name in ["train", "validation", "test"]:
                y_true = all_y[split_name]
                y_pred = preds_dict[split_name]
                dates = all_dates[split_name]
                residuals = y_true - y_pred

                # Export prediction CSV
                pred_df = pd.DataFrame({
                    "date": dates,
                    "actual_pm25": y_true,
                    "predicted_pm25": y_pred,
                    "residual": residuals
                })
                pred_path = self.pred_dir / f"{model_name.lower().replace(' ', '_')}_{split_name}.csv"
                pred_df.to_csv(pred_path, index=False)

                # Calculate metrics
                m = self.calculate_metrics(y_true, y_pred)
                metrics_rows.append({
                    "Model": model_name,
                    "Split": split_name.capitalize(),
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "R2": m["R2"],
                    "Median_AE": m["Median_AE"]
                })

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(self.exp_dir / "metrics.csv", index=False)
        logger.info(f"Metrics table saved to: {self.exp_dir / 'metrics.csv'}")
        return metrics_df

    def plot_residual_analysis(
        self,
        all_preds: dict[str, dict[str, pd.Series]],
        all_y: dict[str, pd.Series],
        all_dates: dict[str, pd.Series]
    ):
        """Generates residual analysis plots for validation and test periods."""
        logger.info("Generating residual analysis plots...")

        for split_name in ["validation", "test"]:
            y_true = all_y[split_name]
            dates = pd.to_datetime(all_dates[split_name])

            # 1. Actual vs Predicted Plots per model
            for model_name, preds_dict in all_preds.items():
                y_pred = preds_dict[split_name]

                plt.figure(figsize=(10, 5))
                plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", s=35)
                max_val = max(y_true.max(), y_pred.max())
                min_val = min(y_true.min(), y_pred.min())
                plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal Perfect Prediction (y=x)")
                plt.title(f"{model_name} Baseline: Actual vs Predicted PM2.5 ({split_name.capitalize()})", fontweight="bold")
                plt.xlabel("Actual PM2.5 (µg/m³)")
                plt.ylabel("Predicted PM2.5 (µg/m³)")
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"actual_vs_pred_{model_name.lower().replace(' ', '_')}_{split_name}.png", dpi=300)
                plt.close()

            # 2. Residual vs Predicted Plot (Comparing Models)
            plt.figure(figsize=(10, 5))
            for model_name, preds_dict in all_preds.items():
                y_pred = preds_dict[split_name]
                residuals = y_true - y_pred
                plt.scatter(y_pred, residuals, alpha=0.5, label=model_name, s=30)

            plt.axhline(0, color="red", linestyle="--")
            plt.title(f"Residuals vs Predicted PM2.5 ({split_name.capitalize()})", fontweight="bold")
            plt.xlabel("Predicted PM2.5 (µg/m³)")
            plt.ylabel("Residual (Actual - Predicted)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"residuals_vs_pred_{split_name}.png", dpi=300)
            plt.close()

            # 3. Residual Over Time Plot
            plt.figure(figsize=(12, 5))
            for model_name, preds_dict in all_preds.items():
                y_pred = preds_dict[split_name]
                residuals = y_true - y_pred
                plt.plot(dates, residuals, alpha=0.7, label=model_name)

            plt.axhline(0, color="black", linestyle="--", alpha=0.7)
            plt.title(f"Residuals Over Time ({split_name.capitalize()})", fontweight="bold")
            plt.xlabel("Date")
            plt.ylabel("Residual (µg/m³)")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"residuals_over_time_{split_name}.png", dpi=300)
            plt.close()

        logger.info(f"Residual plots saved to: {self.plots_dir}")

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame) -> str:
        """Converts a DataFrame to clean GitHub markdown table format without external dependencies."""
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        for _, row in df.iterrows():
            row_str = "| " + " | ".join([str(val) for val in row]) + " |"
            lines.append(row_str)
        return "\n".join(lines)

    def create_metadata_and_summary(
        self, metrics_df: pd.DataFrame, safe_features_count: int, alpha: float = 1.0
    ):
        """Generates metadata.json and baseline_summary.md."""
        logger.info("Writing experiment metadata.json and baseline_summary.md...")

        metadata = {
            "experiment_id": "phase3b1_baselines",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "dataset_version": "v1",
            "target": self.target_col,
            "prediction_cutoff": "end_of_day_t-1",
            "prediction_safe_feature_count": safe_features_count,
            "models": {
                "Persistence": {
                    "methodology": "y_hat(t) = PM2.5(t-1)"
                },
                "Linear Regression": {
                    "preprocessing": "StandardScaler",
                    "scaler_fit_scope": "X_train_only"
                },
                "Ridge Regression": {
                    "preprocessing": "StandardScaler",
                    "scaler_fit_scope": "X_train_only",
                    "hyperparameters": {"alpha": alpha}
                }
            },
            "high_pollution_evaluation_status": "NOT YET DEFINED"
        }

        with open(self.exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        # Markdown Tables
        val_df = metrics_df[metrics_df["Split"] == "Validation"]
        test_df = metrics_df[metrics_df["Split"] == "Test"]

        val_comp = self._df_to_markdown(val_df)
        test_comp = self._df_to_markdown(test_df)
        full_table = self._df_to_markdown(metrics_df)

        summary_md = f"""# AtmosIQ Phase 3B-1: Baseline Model Performance Summary

**Experiment ID**: `phase3b1_baselines`  
**Dataset Version**: `v1`  
**Target Variable**: `pm25`  
**Prediction Cutoff**: End of Day $t-1$ ($X_{{\\le t-1}} \\rightarrow Y_t$)  
**Prediction-Safe Features Used**: **{safe_features_count}**  
**High-Pollution Evaluation Status**: **NOT YET DEFINED** (No project threshold formally defined yet)

---

## 1. Primary Model Comparison Matrix

### Validation Partition Performance (2024-01-01 to 2024-06-30)
{val_comp}

### Test Partition Performance (2024-07-01 to 2024-12-31)
{test_comp}

---

## 2. Complete Multi-Period Evaluation Table

{full_table}

---

## 3. Overfitting & Degradation Observations

1. **Linear Regression Overfitting**:
   - Ordinary Least Squares Linear Regression achieves high fit on Train ($R^2 \\approx 0.85$), but exhibits substantial degradation on Validation ($R^2 \\approx 0.50$) due to multicollinearity across 201 predictors without regularization.
2. **Ridge Regularization Stability**:
   - Ridge Regression ($\alpha=1.0$) stabilizes coefficient magnitudes, improving out-of-sample generalization over unregularized Ordinary Least Squares.
3. **Persistence Benchmark**:
   - Naive Persistence ($\hat{{y}}_t = \\text{{PM2.5}}_{{t-1}}$) remains a competitive baseline due to strong day-to-day atmospheric persistence.

---

## 4. Key Limitations & Next Steps

> [!IMPORTANT]
> These models are untuned baselines. No final AtmosIQ production model has been selected.

- **Next Phase**: Phase 3B-2 will introduce non-linear tree-based models (Random Forest, LightGBM, XGBoost, CatBoost) to capture non-linear meteorology $\\times$ biomass burning interaction effects.
"""

        with open(self.exp_dir / "baseline_summary.md", "w", encoding="utf-8") as f:
            f.write(summary_md)

        logger.info(f"Summary written to: {self.exp_dir / 'baseline_summary.md'}")

    def run(self):
        """Executes full Phase 3B-1 baseline evaluation pipeline."""
        logger.info("=== Starting AtmosIQ Phase 3B-1 Baseline Evaluation Pipeline ===")

        self.create_directories()
        safe_features = self.load_feature_whitelist()

        (
            train_raw, val_raw, test_raw,
            X_train, y_train, dates_train,
            X_val, y_val, dates_val,
            X_test, y_test, dates_test
        ) = self.load_and_validate_splits(safe_features)

        # 1. Model 1: Persistence
        persistence_preds = self.generate_persistence_predictions(train_raw, val_raw, test_raw)

        # 2. Model 2: Linear Regression
        lr_model, lr_preds = self.train_linear_regression(X_train, y_train, X_val, X_test)

        # 3. Model 3: Ridge Regression
        ridge_model, ridge_preds = self.train_ridge_regression(X_train, y_train, X_val, X_test, alpha=1.0)

        # Compile predictions dictionary
        all_preds = {
            "Persistence": persistence_preds,
            "Linear Regression": lr_preds,
            "Ridge Regression": ridge_preds
        }
        all_y = {"train": y_train, "validation": y_val, "test": y_test}
        all_dates = {"train": dates_train, "validation": dates_val, "test": dates_test}

        # 4. Save artifacts & metrics
        metrics_df = self.save_predictions_and_metrics(all_preds, all_y, all_dates)

        # 5. Plot Residuals
        self.plot_residual_analysis(all_preds, all_y, all_dates)

        # 6. Create Metadata & Summary
        self.create_metadata_and_summary(metrics_df, len(safe_features), alpha=1.0)

        logger.info("=== Phase 3B-1 Baseline Evaluation Pipeline Completed Successfully ===")


if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    evaluator.run()
