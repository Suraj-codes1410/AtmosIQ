package com.atmosiq.tools.forecast;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Strongly typed input schema for the forecast_pm25 tool.
 * Documented for Spring AI Function Calling and tool schema generation.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ForecastToolRequest {

    @JsonProperty(required = true)
    @JsonPropertyDescription("Sequential list of daily environmental observation records (minimum 14 rows) containing the 35 prediction-safe features")
    private List<Map<String, Object>> records;

    @JsonPropertyDescription("Optional correlation ID for distributed tracing")
    private String correlationId;
}
