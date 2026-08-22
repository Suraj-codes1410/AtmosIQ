package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * End-to-end request & model provenance tracking metadata.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ProvenanceMetadataDto {

    private String requestId;
    private String correlationId;
    private String timestampUtc;
    private String releaseVersion;
    private String modelId;
    private String modelSha256;
    private String toolName;
    private String downstreamService;
    private Long latencyMs;
}
