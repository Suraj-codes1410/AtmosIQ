package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Top-level application forecast response returned to clients.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ForecastResponseDto {

    private String status;
    private String modelVersion;
    private Integer forecastCount;
    private List<ForecastItemDto> forecasts;
    private ModelMetadataDto modelMetadata;
    private ProvenanceMetadataDto provenance;
}
