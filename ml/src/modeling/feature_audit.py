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
from ml.src.utils.logger import setup_logger

logger = setup_logger("FeatureAuditPhase3C")


class FeatureAuditEngine:
    """
    AtmosIQ Phase 3C: Feature Integrity & Audit Engine.
    Performs comprehensive feature classification, summary statistics calculation,
    low-variance filtering, and pairwise correlation redundancy analysis strictly using TRAIN data.
    """

    def __init__(
        self,
        modeling_dir: str = "ml/data/modeling/v1",
        exp_dir: str = "ml/experiments/phase3c"
    ):
        self.modeling_dir = Path(modeling_dir)
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = "pm25"
        self.date_col = "date"

    @staticmethod
    def load_safe_features() -> tuple[pd.DataFrame, list[str]]:
        """Loads feature_availability.csv and returns authoritative safe feature whitelist (201 features)."""
        avail_path = Path("ml/data/modeling/v1/feature_availability.csv")
        assert avail_path.exists(), f"Feature availability file missing: {avail_path}"

        df_avail = pd.read_csv(avail_path)
        safe_df = df_avail[df_avail["prediction_safe"] == True]
        safe_features = safe_df["feature_name"].tolist()

        assert len(safe_features) == 201, f"Expected 201 safe features, got {len(safe_features)}"
        assert "date" not in safe_features
        assert "pm25" not in safe_features

        return df_avail, safe_features

    @staticmethod
    def classify_feature_group(col_name: str) -> str:
        """Classifies a feature into an environmental process category."""
        c = col_name.lower()

        # 1. PM2.5 Historical
        if c.startswith("pm25_lag_") or c.startswith("pm25_roll_"):
            return "pm25_historical"

        # 2. Other Pollutant Historical
        if any(c.startswith(prefix) for prefix in [
            "pm10_lag_", "pm10_roll_", "no2_lag_", "no2_roll_", "so2_lag_",
            "co_normalized", "no2_so2_ratio", "pm25_pm10_ratio", "daily_pollutant_trend",
            "pollutant_rolling_avg", "pollutant_volatility", "pollutant_zscore", "pollutant_anomaly_score"
        ]):
            return "other_pollutant_historical"

        # 3. Wind / Ventilation
        if any(w in c for w in ["wind_speed", "wind_x", "wind_y", "wind_direction", "pressure_x_wind"]):
            return "wind_ventilation"

        # 4. Fire / Biomass Burning
        if any(f in c for f in [
            "fire_", "brightness", "stubble", "transport_score", "distance_weighted"
        ]):
            return "fire_biomass_burning"

        # 5. Meteorological
        if any(m in c for m in [
            "temp", "temperature", "humidity", "pressure", "precip", "rain", "dry_day", "thi"
        ]):
            return "meteorological"

        # 6. Calendar / Seasonal
        if col_name in [
            "day_of_week", "month", "quarter", "day_of_year", "week_of_year",
            "is_weekend", "is_winter", "is_summer", "is_monsoon", "is_post_monsoon",
            "is_stubble_season", "days_until_diwali", "days_since_diwali",
            "festival_window", "traffic_activity_proxy", "is_holiday", "is_festival"
        ]:
            return "calendar_seasonal"

        # 7. Interaction / Derived
        if any(x in c for x in ["_x_", "interaction"]):
            return "interaction_derived"

        return "other"

    def run_feature_registry_audit(self, df_train: pd.DataFrame, safe_features: list[str]) -> pd.DataFrame:
        """Calculates summary statistics for all 201 safe features strictly on TRAIN set."""
        logger.info(f"Auditing summary statistics for {len(safe_features)} safe features on train.csv...")

        records = []
        for col in safe_features:
            s = df_train[col]
            group = self.classify_feature_group(col)

            records.append({
                "feature_name": col,
                "feature_group": group,
                "dtype": str(s.dtype),
                "missing_count": int(s.isnull().sum()),
                "missing_rate": float(s.isnull().mean()),
                "infinite_count": int(np.isinf(s.values).sum()),
                "unique_count": int(s.nunique()),
                "variance": round(float(s.var()), 6),
                "min": round(float(s.min()), 6),
                "max": round(float(s.max()), 6),
                "mean": round(float(s.mean()), 6),
                "median": round(float(s.median()), 6),
                "standard_deviation": round(float(s.std()), 6),
                "prediction_safe": True,
                "source_module": "ml.src.features"
            })

        registry_df = pd.DataFrame(records)
        registry_path = self.exp_dir / "feature_registry.csv"
        registry_df.to_csv(registry_path, index=False)
        logger.info(f"Feature registry written to: {registry_path}")

        # Feature Group Summary Table
        group_summary = registry_df.groupby("feature_group").agg(
            feature_count=("feature_name", "count"),
            mean_variance=("variance", "mean"),
            zero_variance_count=("variance", lambda v: (v == 0).sum())
        ).reset_index()
        group_summary.to_csv(self.exp_dir / "feature_group_summary.csv", index=False)

        return registry_df

    def audit_duplicates_and_low_variance(self, df_train: pd.DataFrame, registry_df: pd.DataFrame):
        """Identifies exact duplicates, constant features, and low variance features on TRAIN set."""
        logger.info("Auditing duplicate and low-variance features on train.csv...")

        # 1. Constant / Zero-variance features
        zero_var_df = registry_df[registry_df["variance"] == 0.0].copy()
        zero_var_df.to_csv(self.exp_dir / "duplicate_features.csv", index=False)

        # 2. Low-variance features (variance < 1e-4 or unique <= 2 non-binary)
        low_var_df = registry_df[(registry_df["variance"] < 1e-4) | (registry_df["unique_count"] <= 1)].copy()
        low_var_df.to_csv(self.exp_dir / "low_variance_features.csv", index=False)

        logger.info(f"Zero-variance features found: {len(zero_var_df)}, Low-variance features: {len(low_var_df)}")

    def audit_correlation_redundancy(
        self, df_train: pd.DataFrame, safe_features: list[str], threshold: float = 0.95
    ) -> pd.DataFrame:
        """Calculates pairwise Pearson & Spearman correlation matrix on TRAIN set and identifies pairs with |r| >= 0.95."""
        logger.info(f"Computing pairwise correlation redundancy (|r| >= {threshold}) on train.csv...")

        X_tr = df_train[safe_features]
        pearson_corr = X_tr.corr(method="pearson").abs()
        spearman_corr = X_tr.corr(method="spearman").abs()

        cols = safe_features
        n = len(cols)

        pairs = []
        redundancy_clusters = {}

        for i in range(n):
            col_a = cols[i]
            group_a = self.classify_feature_group(col_a)

            for j in range(i + 1, n):
                col_b = cols[j]
                group_b = self.classify_feature_group(col_b)

                p_corr = float(pearson_corr.loc[col_a, col_b])
                s_corr = float(spearman_corr.loc[col_a, col_b])

                if p_corr >= threshold or s_corr >= threshold:
                    # Choose proposed representative based on standard domain rule:
                    rep = col_a if len(col_a) <= len(col_b) else col_b
                    reason = f"High pairwise correlation (Pearson: {p_corr:.4f}, Spearman: {s_corr:.4f})"

                    pairs.append({
                        "feature_A": col_a,
                        "feature_B": col_b,
                        "pearson_correlation": round(p_corr, 4),
                        "spearman_correlation": round(s_corr, 4),
                        "feature_group_A": group_a,
                        "feature_group_B": group_b,
                        "proposed_representative": rep,
                        "reason": reason
                    })

                    redundancy_clusters.setdefault(rep, []).append(col_b if rep == col_a else col_a)

        pairs_df = pd.DataFrame(pairs)
        pairs_df.to_csv(self.exp_dir / "correlation_pairs.csv", index=False)
        logger.info(f"Correlation pairs (|r| >= {threshold}) saved to: {self.exp_dir / 'correlation_pairs.csv'}. Total pairs: {len(pairs_df)}")

        # Build Redundancy Groups Summary
        group_rows = []
        for rep, redundant_list in redundancy_clusters.items():
            group_rows.append({
                "representative_feature": rep,
                "feature_group": self.classify_feature_group(rep),
                "redundant_feature_count": len(set(redundant_list)),
                "redundant_features": ", ".join(sorted(list(set(redundant_list))))
            })

        red_df = pd.DataFrame(group_rows)
        red_df.to_csv(self.exp_dir / "redundancy_groups.csv", index=False)
        logger.info(f"Redundancy groups summary saved to: {self.exp_dir / 'redundancy_groups.csv'}")

        return pairs_df

    def run(self):
        """Executes full Phase 3C feature audit pipeline."""
        logger.info("=== Starting Phase 3C Feature Integrity & Audit Engine ===")

        train_path = self.modeling_dir / "train.csv"
        assert train_path.exists(), f"Train file missing: {train_path}"
        df_train = pd.read_csv(train_path)

        df_avail, safe_features = self.load_safe_features()

        registry_df = self.run_feature_registry_audit(df_train, safe_features)
        self.audit_duplicates_and_low_variance(df_train, registry_df)
        self.audit_correlation_redundancy(df_train, safe_features, threshold=0.95)

        logger.info("=== Phase 3C Feature Audit Engine Completed Successfully ===")


if __name__ == "__main__":
    auditor = FeatureAuditEngine()
    auditor.run()
