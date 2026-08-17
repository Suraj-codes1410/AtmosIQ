"""
AtmosIQ Phase 8E: Comprehensive Deep-Learning Readiness Benchmark Engine.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from .config import Phase8EConfig
from .dataset_loader import Phase8ETemporalDataLoader
from .models import TemporalModelBenchmarkEngine

logger = logging.getLogger(__name__)


class Phase8EBenchmarkRunner:
    """Executes multi-architecture, multi-augmentation, multi-seed benchmarks on the locked 2022-2024 fold."""

    def __init__(self, config: Phase8EConfig, feature_registry: List[str]):
        self.config = config
        self.feature_registry = list(feature_registry)
        self.loader = Phase8ETemporalDataLoader(self.feature_registry)

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df_test_eval: pd.DataFrame,
        meta_info: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Computes standard, extreme-event, and temporal robustness metrics."""
        # 1. Standard Metrics
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        pr, _ = pearsonr(y_true, y_pred) if len(y_true) > 2 else (0.0, 0.0)
        bias = float(np.mean(y_pred - y_true))

        std_metrics = {
            **meta_info,
            "test_samples": len(y_true),
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "pearson_r": float(pr),
            "mean_bias": bias,
        }

        # 2. Extreme-Event Metrics (PM2.5 >= 250)
        ext_mask = (y_true >= 250.0)
        ext_count = int(ext_mask.sum())
        if ext_count > 0:
            ext_mae = float(mean_absolute_error(y_true[ext_mask], y_pred[ext_mask]))
            ext_rmse = float(np.sqrt(mean_squared_error(y_true[ext_mask], y_pred[ext_mask])))
            ext_bias = float(np.mean(y_pred[ext_mask] - y_true[ext_mask]))
        else:
            ext_mae, ext_rmse, ext_bias = 0.0, 0.0, 0.0

        ext_metrics = {
            **meta_info,
            "extreme_samples": ext_count,
            "extreme_mae": ext_mae,
            "extreme_rmse": ext_rmse,
            "extreme_bias": ext_bias,
        }

        # 3. Temporal Robustness by Year, Season, Regime
        df_eval_copy = df_test_eval.copy()
        df_eval_copy["y_true"] = y_true
        df_eval_copy["y_pred"] = y_pred

        temporal_records = []
        # By Year
        for yr, df_grp in df_eval_copy.groupby("year"):
            temporal_records.append({
                **meta_info,
                "dimension": "Year",
                "category": str(yr),
                "samples": len(df_grp),
                "mae": float(mean_absolute_error(df_grp["y_true"], df_grp["y_pred"])),
                "rmse": float(np.sqrt(mean_squared_error(df_grp["y_true"], df_grp["y_pred"]))),
            })
        # By Season
        for s, df_grp in df_eval_copy.groupby("season"):
            temporal_records.append({
                **meta_info,
                "dimension": "Season",
                "category": str(s),
                "samples": len(df_grp),
                "mae": float(mean_absolute_error(df_grp["y_true"], df_grp["y_pred"])),
                "rmse": float(np.sqrt(mean_squared_error(df_grp["y_true"], df_grp["y_pred"]))),
            })
        # By Regime
        for reg, df_grp in df_eval_copy.groupby("pollution_regime"):
            temporal_records.append({
                **meta_info,
                "dimension": "Regime",
                "category": str(reg),
                "samples": len(df_grp),
                "mae": float(mean_absolute_error(df_grp["y_true"], df_grp["y_pred"])),
                "rmse": float(np.sqrt(mean_squared_error(df_grp["y_true"], df_grp["y_pred"]))),
            })

        return std_metrics, ext_metrics, temporal_records

    def run_all_benchmarks(
        self,
        df_real_train: pd.DataFrame,
        df_real_test: pd.DataFrame,
        df_8c_corpus: pd.DataFrame,
        df_8d_corpus: pd.DataFrame
    ) -> Dict[str, Any]:
        """Runs complete benchmark suite across configurations, architectures, seeds, and horizons."""
        self.loader.fit_scaler(df_real_train)

        benchmark_rows = []
        extreme_rows = []
        temporal_rows = []
        seed_rows = []

        window_size = self.config.default_sequence_window
        d_feat = len([f for f in self.feature_registry if f in df_real_train.columns])

        # Prepare test evaluation frame matching the sequence output length
        X_test, y_test = self.loader.create_sequences(df_real_test, window_size=window_size, is_synthetic=False)
        df_test_eval = df_real_test.iloc[window_size:].copy().reset_index(drop=True)

        logger.info(f"Test evaluation sequences prepared: {len(X_test)} samples (window={window_size}).")

        # 1. Main Architecture & Configuration Benchmark (using primary_seed=42)
        for cfg in self.config.configurations:
            cfg_id = cfg["id"]
            cfg_name = cfg["name"]
            corpus_type = cfg["corpus"]
            ratio = cfg["ratio"]

            # Select synthetic dataframe
            if corpus_type == "8C":
                df_synth = df_8c_corpus
            elif corpus_type == "8D":
                df_synth = df_8d_corpus
            else:
                df_synth = None

            for arch in self.config.architectures:
                meta = {
                    "config_id": cfg_id,
                    "config_name": cfg_name,
                    "architecture": arch,
                    "augmentation_ratio": ratio,
                    "corpus_type": corpus_type,
                    "seed": self.config.primary_seed,
                    "window_size": window_size,
                }
                logger.info(f"Running Benchmark: Config={cfg_id}, Arch={arch}, Seed={self.config.primary_seed}...")

                X_tr, y_tr = self.loader.build_augmented_training_set(
                    df_real_train, df_synth, augmentation_ratio=ratio, window_size=window_size, seed=self.config.primary_seed
                )

                model = TemporalModelBenchmarkEngine.get_model(arch, window_size, d_feat, self.config.primary_seed)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_test)

                std_res, ext_res, temp_res = self.evaluate_predictions(y_test, preds, df_test_eval, meta)
                std_res["train_samples"] = len(X_tr)
                benchmark_rows.append(std_res)
                extreme_rows.append(ext_res)
                temporal_rows.extend(temp_res)

        # 2. Multi-Seed Statistical Reproducibility Benchmark (Primary 25% comparison across seeds)
        for seed in self.config.seeds:
            for cfg_id, corpus_type, ratio in [
                ("REAL_ONLY", "none", 0.0),
                ("REAL_PLUS_8C_25", "8C", 0.25),
                ("REAL_PLUS_8D_25", "8D", 0.25),
            ]:
                df_synth = df_8c_corpus if corpus_type == "8C" else (df_8d_corpus if corpus_type == "8D" else None)
                for arch in ["LSTM", "TCN", "Transformer"]:
                    X_tr, y_tr = self.loader.build_augmented_training_set(
                        df_real_train, df_synth, augmentation_ratio=ratio, window_size=window_size, seed=seed
                    )
                    model = TemporalModelBenchmarkEngine.get_model(arch, window_size, d_feat, seed)
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_test)

                    mae = float(mean_absolute_error(y_test, preds))
                    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                    r2 = float(r2_score(y_test, preds))

                    seed_rows.append({
                        "config_id": cfg_id,
                        "architecture": arch,
                        "seed": seed,
                        "test_mae": mae,
                        "test_rmse": rmse,
                        "test_r2": r2,
                    })

        df_benchmarks = pd.DataFrame(benchmark_rows)
        df_extremes = pd.DataFrame(extreme_rows)
        df_temporals = pd.DataFrame(temporal_rows)
        df_seeds = pd.DataFrame(seed_rows)

        # 3. Candidate Ranking & Selection Matrix
        df_agg = df_benchmarks.groupby("config_id").agg({
            "test_mae": "mean",
            "test_rmse": "mean",
            "test_r2": "mean",
            "pearson_r": "mean",
        }).reset_index()

        df_ext_agg = df_extremes.groupby("config_id").agg({
            "extreme_mae": "mean",
            "extreme_rmse": "mean",
        }).reset_index()

        df_ranking = pd.merge(df_agg, df_ext_agg, on="config_id")

        # Multi-objective composite score (lower is better)
        mae_ptp = np.ptp(df_ranking["test_mae"].values)
        ext_ptp = np.ptp(df_ranking["extreme_mae"].values)
        mae_norm = (df_ranking["test_mae"] - df_ranking["test_mae"].min()) / (mae_ptp + 1e-6)
        ext_norm = (df_ranking["extreme_mae"] - df_ranking["extreme_mae"].min()) / (ext_ptp + 1e-6)
        df_ranking["composite_score"] = 0.60 * mae_norm + 0.40 * ext_norm
        df_ranking = df_ranking.sort_values("composite_score").reset_index(drop=True)
        df_ranking["rank"] = np.arange(1, len(df_ranking) + 1)

        return {
            "df_benchmarks": df_benchmarks,
            "df_extremes": df_extremes,
            "df_temporals": df_temporals,
            "df_seeds": df_seeds,
            "df_ranking": df_ranking,
        }
