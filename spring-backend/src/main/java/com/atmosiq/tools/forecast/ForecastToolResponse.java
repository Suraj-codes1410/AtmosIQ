package com.atmosiq.tools.forecast;

import com.atmosiq.model.response.ForecastItemDto;
import com.atmosiq.model.response.ModelMetadataDto;
import com.atmosiq.model.response.ProvenanceMetadataDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Output schema returned by the forecast_pm25 tool.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ForecastToolResponse {

    private String status;
    private String modelVersion;
    private Integer forecastCount;
    private List<ForecastItemDto> forecasts;
    private ModelMetadataDto modelMetadata;
    private ProvenanceMetadataDto provenance;
}
