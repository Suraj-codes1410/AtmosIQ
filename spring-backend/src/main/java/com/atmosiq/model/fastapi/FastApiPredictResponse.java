package com.atmosiq.model.fastapi;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FastApiPredictResponse {

    @JsonProperty("status")
    private String status;

    @JsonProperty("model_version")
    private String modelVersion;

    @JsonProperty("execution_latency_ms")
    private Double executionLatencyMs;

    @JsonProperty("batch_size")
    private Integer batchSize;

    @JsonProperty("forecasts")
    private List<FastApiForecastItem> forecasts;
}
