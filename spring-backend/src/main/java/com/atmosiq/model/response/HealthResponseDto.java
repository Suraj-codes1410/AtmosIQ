package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class HealthResponseDto {

    private String status;
    private String orchestrationLayer;
    private String downstreamService;
    private Boolean downstreamModelLoaded;
    private String timestampUtc;
    private String correlationId;
}
