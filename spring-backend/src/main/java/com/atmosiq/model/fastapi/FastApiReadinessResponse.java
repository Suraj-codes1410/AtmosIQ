package com.atmosiq.model.fastapi;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FastApiReadinessResponse {

    @JsonProperty("status")
    private String status;

    @JsonProperty("model_version")
    private String modelVersion;

    @JsonProperty("feature_count")
    private Integer featureCount;

    @JsonProperty("scaler_ready")
    private Boolean scalerReady;

    @JsonProperty("calibration_ready")
    private Boolean calibrationReady;

    @JsonProperty("timestamp_utc")
    private String timestampUtc;
}
