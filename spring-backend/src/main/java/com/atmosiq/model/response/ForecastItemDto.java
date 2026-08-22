package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ForecastItemDto {

    private String predictionId;
    private String timestampUtc;
    private Double forecastPm25;
    private UncertaintyDto uncertainty;
}
