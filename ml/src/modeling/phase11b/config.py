"""
AtmosIQ Phase 11B: Configuration & Production Invariants.
"""

from pathlib import Path
from dataclasses import dataclass

CERTIFIED_RELEASE_ID       = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0"
CERTIFIED_CANDIDATE_ID     = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0"
CERTIFIED_MODEL_SHA256     = "fdc99f7ca4410f3d577e52718e35c956c97f368cd91a8b7f505ee23824085bac"
CERTIFIED_ARCHITECTURE     = "TCN"
CERTIFIED_PARAMS           = 849
CERTIFIED_WINDOW           = 14
CERTIFIED_FEATURE_DIM      = 35
CERTIFIED_AUGMENTATION     = "25% CAL-07"
CERTIFIED_GIT_TAG          = "v1.0.0"
CERTIFIED_PROTECTED_COUNT  = 34
CERTIFIED_CALIBRATION_BIAS = -5.06   # µg/m³
CERTIFIED_BOUND_90         = 95.66   # µg/m³
FALLBACK_TARGET            = "MODEL_V3_PRODUCTION"

# SLA Limits
SLA_SINGLE_INFERENCE_MS    = 10.0
SLA_BATCH_PIPELINE_MS      = 50.0
SLA_MAX_MEMORY_MB          = 256.0
SLA_MIN_THROUGHPUT_SPS     = 100.0

# Drift Thresholds (Reused from Phase 10B)
PSI_GREEN_THRESHOLD        = 0.10
PSI_YELLOW_THRESHOLD       = 0.25
WASSERSTEIN_GREEN_MAX      = 0.50

PRODUCTION_FEATURES = [
    "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d",
    "pm25_roll_mean_3d", "pm25_roll_mean_7d", "pm25_roll_mean_14d",
    "pm25_roll_std_7d", "pm25_roll_max_7d", "pm25_roll_min_7d",
    "temperature_c_lag_1d", "temperature_c_roll_mean_3d", "temperature_c_roll_min_3d",
    "humidity_pct_lag_1d", "humidity_pct_roll_mean_3d", "humidity_pct_roll_max_7d",
    "wind_speed_kmh_lag_1d", "wind_speed_kmh_roll_mean_3d",
    "wind_u_component_1d", "wind_v_component_1d",
    "is_stubble_season", "fire_hotspot_count_lag_1d",
    "fire_hotspot_count_roll_mean_3d", "fire_hotspot_count_roll_mean_7d",
    "upwind_stubble_quadrant_1d",
    "rainfall_1d", "rainfall_3d", "rain_event_1d", "washout_index_3d",
    "pblh_1d", "pblh_min_1d", "pblh_roll_mean_3d",
    "ventilation_index_1d", "aod_550_1d", "festival_window",
]


@dataclass(frozen=True)
class Phase11BConfig:
    """Immutable configuration for Phase 11B."""
    root_dir: Path
    experiments_dir: Path = Path("ml/experiments/phase11b_monitoring")
    reports_dir: Path = Path("ml/experiments/phase11b_monitoring/reports")
    manifests_dir: Path = Path("ml/experiments/phase11b_monitoring/manifests")
    data_dir: Path = Path("ml/experiments/phase11b_monitoring/data")
    figures_dir: Path = Path("ml/experiments/phase11b_monitoring/figures")
    bundle_dir: Path = Path("ml/experiments/phase10d_release/release_bundle")
    dataset_path: Path = Path("ml/data/modeling/v3/feature_dataset_frozen.csv")
    phase10e_hash_manifest: Path = Path("ml/experiments/phase10e_certification/hashes/phase10e_protected_artifacts_post_sha256.json")
    observability_dir: Path = Path("ml/experiments/phase10b_observability/manifests")
