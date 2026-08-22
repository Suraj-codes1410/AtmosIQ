package com.atmosiq.model.fastapi;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Raw payload sent to FastAPI /predict endpoint.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FastApiPredictRequest {

    @JsonProperty("records")
    private List<Map<String, Object>> records;
}
