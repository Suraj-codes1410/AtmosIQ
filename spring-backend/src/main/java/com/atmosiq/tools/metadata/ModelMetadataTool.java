package com.atmosiq.tools.metadata;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.model.fastapi.FastApiVersionResponse;
import com.atmosiq.model.response.ModelMetadataDto;
import com.atmosiq.tools.ToolContract;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.function.Function;

@Slf4j
@Component("getModelMetadataTool")
@RequiredArgsConstructor
public class ModelMetadataTool implements ToolContract<Map<String, Object>, ModelMetadataDto>,
        Function<Map<String, Object>, ModelMetadataDto> {

    public static final String TOOL_NAME = "get_model_metadata";
    public static final String TOOL_DESCRIPTION = "Retrieves architectural and cryptographic metadata for the certified AtmosIQ v1.0.0 TCN model.";

    private final FastApiInferenceClient inferenceClient;
    private final AtmosIQProperties properties;

    @Override
    public String getName() {
        return TOOL_NAME;
    }

    @Override
    public String getDescription() {
        return TOOL_DESCRIPTION;
    }

    @Override
    public ModelMetadataDto apply(Map<String, Object> unused) {
        return execute(unused);
    }

    @Override
    public ModelMetadataDto execute(Map<String, Object> unused) {
        log.debug("Executing ModelMetadataTool");
        FastApiVersionResponse versionResp = inferenceClient.getVersion();

        return ModelMetadataDto.builder()
                .modelId(versionResp.getModelId())
                .candidateId(versionResp.getCandidateId())
                .architecture(versionResp.getArchitecture())
                .parameterCount(versionResp.getParameters())
                .sequenceWindow(14)
                .featureDimension(35)
                .augmentationPolicy("25% CAL-07 (100% Synthetic Strictly Prohibited)")
                .modelSha256(properties.getFastApi().getExpectedModelSha256())
                .releaseStatus("FINAL_PRODUCTION_CERTIFIED")
                .fallbackTarget("MODEL_V3_PRODUCTION")
                .build();
    }
}
