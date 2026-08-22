package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Model architectural metadata & governance state.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ModelMetadataDto {

    private String modelId;
    private String candidateId;
    private String architecture;
    private Integer parameterCount;
    private Integer sequenceWindow;
    private Integer featureDimension;
    private String augmentationPolicy;
    private String modelSha256;
    private String releaseStatus;
    private String fallbackTarget;
}
