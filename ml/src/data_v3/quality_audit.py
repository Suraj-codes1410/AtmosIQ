import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("QualityAuditV3")


class QualityAuditV3:
    """
    Automated External Data Quality Audit Module.
    Generates external_data_quality_report.csv and external_data_quality_report.md.
    """

    def run_audit(self, df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        logger.info("Executing Automated Quality Audit on Processed External Features...")
        results = []

        feature_cols = [c for c in df.columns if c != 'date']

        for col in feature_cols:
            series = df[col]
            null_count = series.isnull().sum()
            inf_count = np.isinf(series).sum()
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()
            std_val = series.std()

            status = "PASS"
            flag_reason = "Clean"

            if null_count > 0:
                status = "FAIL"
                flag_reason = f"Contains {null_count} missing values"
            elif inf_count > 0:
                status = "FAIL"
                flag_reason = f"Contains {inf_count} infinite values"
            elif "rainfall" in col and min_val < 0.0:
                status = "FAIL"
                flag_reason = f"Negative rainfall detected ({min_val})"
            elif "pblh" in col and min_val < 50.0:
                status = "WARNING"
                flag_reason = f"Unusually low PBLH ({min_val}m)"

            results.append({
                "feature_name": col,
                "null_count": int(null_count),
                "inf_count": int(inf_count),
                "min_value": float(min_val),
                "max_value": float(max_val),
                "mean_value": float(mean_val),
                "std_value": float(std_val),
                "audit_status": status,
                "audit_note": flag_reason
            })

        report_df = pd.DataFrame(results)
        csv_path = output_dir / "external_data_quality_report.csv"
        report_df.to_csv(csv_path, index=False)
        logger.info(f"Quality Audit Report CSV saved to {csv_path}.")

        # Generate Markdown Report
        md_path = output_dir / "external_data_quality_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# AtmosIQ Phase 4G — External Data Quality Audit Report\n\n")
            f.write("| Feature | Null Count | Inf Count | Min | Max | Mean | Std | Audit Status | Audit Note |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for _, r in report_df.iterrows():
                f.write(f"| `{r['feature_name']}` | {r['null_count']} | {r['inf_count']} | {r['min_value']:.2f} | {r['max_value']:.2f} | {r['mean_value']:.2f} | {r['std_value']:.2f} | **{r['audit_status']}** | {r['audit_note']} |\n")

        logger.info(f"Quality Audit Report Markdown saved to {md_path}.")

        fail_count = (report_df['audit_status'] == 'FAIL').sum()
        assert fail_count == 0, f"Quality Audit Failed: {fail_count} features failed quality checks!"

        return report_df
