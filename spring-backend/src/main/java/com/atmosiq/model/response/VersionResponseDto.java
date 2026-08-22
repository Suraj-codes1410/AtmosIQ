package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VersionResponseDto {

    private String modelId;
    private String candidateId;
    private String architecture;
    private Integer parameterCount;
    private String modelSha256;
    private String releaseStatus;
    private String releaseVersion;
    private Boolean verifiedProductionIdentity;
}
