package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ReadinessResponseDto {

    private String status;
    private String modelVersion;
    private Integer featureCount;
    private Boolean scalerReady;
    private Boolean calibrationReady;
    private String timestampUtc;
    private String correlationId;
}
