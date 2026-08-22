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
public class FastApiForecastItem {

    @JsonProperty("prediction_id")
    private String predictionId;

    @JsonProperty("timestamp_utc")
    private String timestampUtc;

    @JsonProperty("forecast_pm25")
    private Double forecastPm25;

    @JsonProperty("lower_90")
    private Double lower90;

    @JsonProperty("upper_90")
    private Double upper90;

    @JsonProperty("conformal_half_width")
    private Double conformalHalfWidth;
}
