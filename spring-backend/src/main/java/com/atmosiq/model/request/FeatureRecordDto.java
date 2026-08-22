package com.atmosiq.model.request;

import java.util.List;

/**
 * Registry of the 35 prediction-safe feature names certified in AtmosIQ v1.0.0.
 */
public final class FeatureRecordDto {

    public static final int REQUIRED_WINDOW_SIZE = 14;
    public static final int REQUIRED_FEATURE_DIM = 35;

    public static final List<String> CERTIFIED_35_FEATURES = List.of(
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
            "ventilation_index_1d", "aod_550_1d", "festival_window"
    );

    private FeatureRecordDto() {
        // utility constant class
    }
}
