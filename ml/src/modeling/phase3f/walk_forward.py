import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.base import clone
from ml.src.utils.logger import setup_logger
from ml.src.modeling.phase3f.models import ModelFactoryPhase3F
from ml.src.modeling.phase3f.evaluation import MetricsEvaluatorPhase3F

logger = setup_logger("WalkForwardPhase3F")


class WalkForwardEnginePhase3F:
    """
    AtmosIQ Phase 3F Walk-Forward Engine.
    Executes expanding-window chronological evaluation across Folds 1 (2022), 2 (2023), and 3 (2024 Holdout).
    """

    def __init__(self, modeling_dir: str = "ml/data/modeling/v2", exp_dir: str = "ml/experiments/phase3f"):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir = self.exp_dir / "predictions"
        self.preds_dir.mkdir(parents=True, exist_ok=True)

        self.frozen_file = self.modeling_dir / "feature_dataset_frozen.csv"
        assert self.frozen_file.exists(), f"Frozen dataset missing: {self.frozen_file}"

    def load_dataset(self) -> pd.DataFrame:
        """Loads Dataset v2 frozen snapshot."""
        df = pd.read_csv(self.frozen_file)
        df["date_dt"] = pd.to_datetime(df["date"])
        return df.sort_values("date_dt").reset_index(drop=True)

    def run_all_experiments(self, feature_groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs all models across all feature groups and 3 walk-forward folds."""
        logger.info("Executing Phase 3F 3-Fold Walk-Forward Evaluation...")
        df = self.load_dataset()
        models = ModelFactoryPhase3F.get_models()

        folds = [
            {"fold": 1, "train_end": "2021-12-31", "eval_start": "2022-01-01", "eval_end": "2022-12-31", "eval_year": 2022, "is_holdout": False},
            {"fold": 2, "train_end": "2022-12-31", "eval_start": "2023-01-01", "eval_end": "2023-12-31", "eval_year": 2023, "is_holdout": False},
            {"fold": 3, "train_end": "2023-12-31", "eval_start": "2024-01-01", "eval_end": "2024-12-31", "eval_year": 2024, "is_holdout": True}
        ]

        metrics_rows = []
        overfit_rows = []

        for f_info in folds:
            f_idx = f_info["fold"]
            yr = f_info["eval_year"]
            is_ho = f_info["is_holdout"]

            tr_sub = df[df["date_dt"] <= f_info["train_end"]].copy()
            ev_sub = df[(df["date_dt"] >= f_info["eval_start"]) & (df["date_dt"] <= f_info["eval_end"])].copy()

            y_tr = tr_sub["pm25"]
            y_ev = ev_sub["pm25"]

            logger.info(f"Fold {f_idx} ({yr}): Train ({len(tr_sub)} rows) -> Eval ({len(ev_sub)} rows) [Holdout={is_ho}]...")

            for g_name, cols in feature_groups.items():
                f_cnt = len(cols)
                X_tr = tr_sub[cols]
                X_ev = ev_sub[cols]

                for m_name, model_inst in models.items():
                    if m_name == "Persistence":
                        # Naive baseline: y_hat(t) = PM2.5(t-1)
                        p_tr = pd.Series(np.vstack([y_tr.iloc[0], y_tr.iloc[:-1].values.reshape(-1, 1)]).ravel())
                        p_ev = pd.Series(np.vstack([y_tr.iloc[-1], y_ev.iloc[:-1].values.reshape(-1, 1)]).ravel())

                        m_tr = MetricsEvaluatorPhase3F.calculate_metrics(y_tr, p_tr)
                        m_ev = MetricsEvaluatorPhase3F.calculate_metrics(y_ev, p_ev)
                    else:
                        model = clone(model_inst)
                        model.fit(X_tr, y_tr)

                        p_tr = pd.Series(model.predict(X_tr))
                        p_ev = pd.Series(model.predict(X_ev))

                        m_tr = MetricsEvaluatorPhase3F.calculate_metrics(y_tr, p_tr)
                        m_ev = MetricsEvaluatorPhase3F.calculate_metrics(y_ev, p_ev)

                    # Export predictions
                    pred_df = pd.DataFrame({
                        "date": ev_sub["date"],
                        "actual_pm25": y_ev.values,
                        "predicted_pm25": p_ev.values,
                        "residual": y_ev.values - p_ev.values
                    })
                    pred_file = self.preds_dir / f"pred_fold{f_idx}_{m_name.lower().replace(' ', '_')}_{g_name}.csv"
                    pred_df.to_csv(pred_file, index=False)

                    metrics_rows.append({
                        "Fold": f_idx,
                        "Eval_Year": yr,
                        "Is_Holdout": is_ho,
                        "Model": m_name,
                        "Feature_Group": g_name,
                        "Feature_Count": f_cnt,
                        "Train_Rows": len(tr_sub),
                        "Eval_Rows": len(ev_sub),
                        "MAE": m_ev["MAE"],
                        "RMSE": m_ev["RMSE"],
                        "R2": m_ev["R2"],
                        "Median_AE": m_ev["Median_AE"]
                    })

                    of_metrics = MetricsEvaluatorPhase3F.calculate_overfitting_metrics(m_tr, m_ev)
                    overfit_rows.append({
                        "Fold": f_idx,
                        "Eval_Year": yr,
                        "Is_Holdout": is_ho,
                        "Model": m_name,
                        "Feature_Group": g_name,
                        "Feature_Count": f_cnt,
                        **of_metrics
                    })

        metrics_df = pd.DataFrame(metrics_rows)
        overfit_df = pd.DataFrame(overfit_rows)

        metrics_df.to_csv(self.exp_dir / "feature_group_metrics.csv", index=False)
        overfit_df.to_csv(self.exp_dir / "overfitting_analysis.csv", index=False)

        logger.info(f"Walk-forward evaluation completed: {len(metrics_df)} experiment records generated.")
        return metrics_df, overfit_df


if __name__ == "__main__":
    from ml.src.modeling.phase3f.feature_groups import FeatureGroupManagerPhase3F
    fg_mgr = FeatureGroupManagerPhase3F()
    groups = fg_mgr.build_feature_groups()

    wf = WalkForwardEnginePhase3F()
    wf.run_all_experiments(groups)
