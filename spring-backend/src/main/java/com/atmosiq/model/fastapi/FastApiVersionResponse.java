package com.atmosiq.model.fastapi;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FastApiVersionResponse {

    @JsonProperty("model_id")
    private String modelId;

    @JsonProperty("candidate_id")
    private String candidateId;

    @JsonProperty("architecture")
    private String architecture;

    @JsonProperty("parameters")
    private Integer parameters;

    @JsonProperty("model_sha256")
    private String modelSha256;

    @JsonProperty("release_status")
    private String releaseStatus;
}
