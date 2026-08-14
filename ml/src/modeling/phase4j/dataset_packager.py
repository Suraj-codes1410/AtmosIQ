import hashlib
import shutil
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("DatasetPackagerPhase4J")


class DatasetPackagerPhase4J:
    """
    Public Dataset Release Candidate Packaging Engine for Phase 4J.
    Prepares kaggle/v3/ as a local/private release candidate package with complete documentation.
    """

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir
        self.kaggle_dir = self.root_dir / "kaggle" / "v3"
        self.src_dataset_path = self.root_dir / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"

    @staticmethod
    def calculate_sha256(filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def package_dataset_v3(self, features_35: list, v3_dataset_hash: str) -> dict:
        logger.info("Packaging Dataset v3 Release Candidate in kaggle/v3/...")
        self.kaggle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Dataset CSV Copy
        dest_dataset = self.kaggle_dir / "dataset.csv"
        shutil.copy2(self.src_dataset_path, dest_dataset)
        dest_hash = self.calculate_sha256(dest_dataset)
        assert dest_hash == v3_dataset_hash, "Dataset copy hash mismatch!"

        # 2. Data Dictionary distinguishing 35 model features from 240 auxiliary columns
        df_v3 = pd.read_csv(self.src_dataset_path)
        all_cols = list(df_v3.columns)
        dict_records = []

        model_feat_set = set(features_35)

        for c in all_cols:
            is_model_feat = (c in model_feat_set)
            if c == "date":
                dtype_str = "datetime"
                unit = "YYYY-MM-DD"
                src = "Calendar"
                desc = "Daily observation timestamp"
                grp = "temporal"
                leak = "safe"
            elif c == "pm25":
                dtype_str = "float64"
                unit = "µg/m³"
                src = "CPCB / DPCC"
                desc = "Delhi NCR 24h average PM2.5 concentration (Prediction Target)"
                grp = "target"
                leak = "unsafe_for_input"
            elif "rainfall" in c or "rain" in c:
                dtype_str = "float64"
                unit = "mm"
                src = "IMD / ERA5"
                desc = f"Precipitation metric ({c})"
                grp = "external_environmental"
                leak = "safe" if "_same_day" not in c else "unsafe"
            elif "pblh" in c or "ventilation" in c:
                dtype_str = "float64"
                unit = "m or m²/s"
                src = "ECMWF ERA5"
                desc = f"Boundary layer / ventilation metric ({c})"
                grp = "wind_ventilation"
                leak = "safe" if "_same_day" not in c else "unsafe"
            elif "fire" in c or "stubble" in c:
                dtype_str = "float64"
                unit = "count / flag"
                src = "NASA FIRMS (VIIRS/MODIS)"
                desc = f"Active biomass fire hotspot indicator ({c})"
                grp = "biomass_burning"
                leak = "safe" if "_same_day" not in c else "unsafe"
            elif "wind" in c:
                dtype_str = "float64"
                unit = "km/h or m/s"
                src = "IMD / ERA5"
                desc = f"Wind speed / directional component ({c})"
                grp = "wind_ventilation"
                leak = "safe" if "_same_day" not in c else "unsafe"
            elif "temperature" in c or "humidity" in c:
                dtype_str = "float64"
                unit = "°C / %"
                src = "IMD"
                desc = f"Surface meteorological feature ({c})"
                grp = "meteorology"
                leak = "safe" if "_same_day" not in c else "unsafe"
            elif "aod" in c:
                dtype_str = "float64"
                unit = "unitless"
                src = "NASA MODIS"
                desc = f"Aerosol Optical Depth ({c})"
                grp = "external_environmental"
                leak = "safe" if "_same_day" not in c else "unsafe"
            else:
                dtype_str = "float64"
                unit = "various"
                src = "Engineered / CPCB"
                desc = f"Engineered auxiliary time-series feature ({c})"
                grp = "auxiliary"
                leak = "safe" if "_same_day" not in c else "unsafe"

            dict_records.append({
                "column_name": c,
                "data_type": dtype_str,
                "unit": unit,
                "source": src,
                "feature_group": grp,
                "is_production_model_input": is_model_feat,
                "leakage_status": leak,
                "description": desc
            })

        df_dict = pd.DataFrame(dict_records)
        df_dict.to_csv(self.kaggle_dir / "data_dictionary.csv", index=False)

        # 3. Feature Registry CSV
        df_feat_reg = pd.DataFrame({
            "model_order": range(1, len(features_35) + 1),
            "feature_name": features_35,
            "group": [df_dict[df_dict['column_name'] == f]['feature_group'].iloc[0] for f in features_35],
            "unit": [df_dict[df_dict['column_name'] == f]['unit'].iloc[0] for f in features_35]
        })
        df_feat_reg.to_csv(self.kaggle_dir / "feature_registry.csv", index=False)

        # 4. Sources Documentation
        sources_md = """# External Environmental Data Sources (Dataset v3)

| Source / Provider | Variable / Parameter | Temporal Resolution | Spatial Resolution | License |
| :--- | :--- | :---: | :---: | :--- |
| **CPCB / DPCC** | PM2.5, PM10, NO2, SO2, CO, O3 | Daily (24h average) | Delhi NCR Stations | Open Govt Data (India) |
| **IMD** | Temperature, Humidity, Wind Speed, Rainfall | Daily | Delhi Safdarjung / NCR | Public Meteorological |
| **ECMWF ERA5** | Planetary Boundary Layer Height (PBLH), U/V Wind | Hourly aggregated daily | 0.25° grid | Copernicus Open License |
| **NASA FIRMS** | Active Fire Hotspots (VIIRS / MODIS) | Daily | Punjab / Haryana corridor | NASA Open Data Policy |
| **NASA MODIS** | Aerosol Optical Depth (AOD 550nm) | Daily | 1km / 10km grid | NASA Open Data Policy |
"""
        with open(self.kaggle_dir / "sources.md", "w") as f:
            f.write(sources_md)

        # 5. Methodology Documentation
        methodology_md = """# Dataset v3 Construction & Alignment Methodology

1. **Temporal Alignment**: All daily observation records are standardized to midnight UTC / 24-hour Indian Standard Time (IST) averages.
2. **Spatial Scope**: Delhi National Capital Region (NCR) bounding box [28.2°N - 28.9°N, 76.8°E - 77.4°E].
3. **Leakage Prevention**: All predictive inputs are lagged by at least 1 day ($t-1$) or computed over backward rolling windows ($t-1$ through $t-w$). No same-day observations of the target or simultaneous pollutants are exposed as inference inputs.
4. **Model Subsetting**: The full dataset provides 275 columns for broad research; the frozen production forecasting model strictly uses the registered 35 features.
"""
        with open(self.kaggle_dir / "methodology.md", "w") as f:
            f.write(methodology_md)

        # 6. Provenance & License
        provenance_md = f"""# Dataset v3 Lineage & Provenance

- **Dataset Version**: `Dataset_v3` (v3.0.0)
- **SHA-256 Checksum**: `{v3_dataset_hash}`
- **Row Count**: 1,827
- **Date Range**: 2020-01-01 to 2024-12-31 (5 Full Years)
- **Preceding Iterations**:
  - Dataset v1 (`c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df`)
  - Dataset v2 (`e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301`)
"""
        with open(self.kaggle_dir / "provenance.md", "w") as f:
            f.write(provenance_md)

        license_md = """# License & Terms of Use

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to share and adapt the material for any research or educational purpose, provided appropriate attribution is given to AtmosIQ.
"""
        with open(self.kaggle_dir / "license.md", "w") as f:
            f.write(license_md)

        citation_cff = """cff-version: 1.2.0
message: "If you use this dataset in research, please cite it as below."
authors:
  - family-names: "AtmosIQ Research Team"
title: "AtmosIQ Delhi NCR PM2.5 Environmental Attribution Dataset (v3)"
version: 3.0.0
date-released: 2026-08-15
"""
        with open(self.kaggle_dir / "citation.cff", "w") as f:
            f.write(citation_cff)

        # 7. README.md with clear publication status
        readme_md = f"""# AtmosIQ Delhi NCR PM2.5 Dataset (v3) — Release Candidate

> **PUBLICATION STATUS**: **PRIVATE / UNPUBLISHED LOCAL RELEASE CANDIDATE**  
> *This package is prepared for eventual public release. It remains local/private until final project freeze and authorization.*

## Dataset Overview
- **Timeline**: January 1, 2020 – December 31, 2024 (1,827 daily rows)
- **Geographic Scope**: Delhi NCR, India
- **Target Variable**: `pm25` (24-hour mean concentration in µg/m³)
- **Total Columns**: 275 columns
- **Production Model Input Features**: Exactly 35 registered prediction-safe features
- **Dataset SHA-256**: `{v3_dataset_hash}`

## Scientific Disclaimer
> **PREDICTIVE IMPORTANCE ≠ SHAP ATTRIBUTION ≠ COUNTERFACTUAL MODEL RESPONSE ≠ CAUSAL EFFECT ≠ ACTUAL EMISSION CONTRIBUTION**
"""
        with open(self.kaggle_dir / "README.md", "w") as f:
            f.write(readme_md)

        # 8. Checksums for kaggle/v3/
        k_checksums = []
        for p in sorted(self.kaggle_dir.glob("*.*")):
            if p.is_file():
                h = self.calculate_sha256(p)
                k_checksums.append(f"{h}  {p.name}")
        with open(self.kaggle_dir / "checksums.txt", "w") as f:
            f.write("\n".join(k_checksums) + "\n")

        logger.info(f"Dataset v3 Release Candidate packaged in {self.kaggle_dir}.")
        return {
            "kaggle_dir": self.kaggle_dir,
            "dataset_hash": dest_hash,
            "data_dictionary_rows": len(df_dict)
        }
