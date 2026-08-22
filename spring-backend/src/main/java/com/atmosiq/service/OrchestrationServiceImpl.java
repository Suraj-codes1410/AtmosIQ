package com.atmosiq.service;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.InvalidForecastRequestException;
import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.model.fastapi.FastApiReadinessResponse;
import com.atmosiq.model.fastapi.FastApiVersionResponse;
import com.atmosiq.model.request.FeatureRecordDto;
import com.atmosiq.model.request.ForecastRequestDto;
import com.atmosiq.model.response.ForecastResponseDto;
import com.atmosiq.model.response.HealthResponseDto;
import com.atmosiq.model.response.ReadinessResponseDto;
import com.atmosiq.model.response.VersionResponseDto;
import com.atmosiq.observability.OrchestrationMetrics;
import com.atmosiq.provenance.ProvenanceTracker;
import com.atmosiq.tools.ToolContract;
import com.atmosiq.tools.ToolRegistry;
import com.atmosiq.tools.forecast.ForecastTool;
import com.atmosiq.tools.forecast.ForecastToolRequest;
import com.atmosiq.tools.forecast.ForecastToolResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.Instant;

@Slf4j
@Service
@RequiredArgsConstructor
public class OrchestrationServiceImpl implements OrchestrationService {

    private final ToolRegistry toolRegistry;
    private final FastApiInferenceClient inferenceClient;
    private final AtmosIQProperties properties;
    private final ProvenanceTracker provenanceTracker;
    private final OrchestrationMetrics metrics;

    @Override
    public ForecastResponseDto executeForecast(ForecastRequestDto request, String correlationId) {
        long startTime = System.currentTimeMillis();
        metrics.recordForecastRequest();

        if (correlationId != null && !correlationId.isBlank()) {
            provenanceTracker.setCurrentCorrelationId(correlationId);
        }
        String effectiveCorrelationId = provenanceTracker.getCurrentCorrelationId();

        log.info("Orchestrating forecast request [CorrelationId: {}, Records: {}]",
                effectiveCorrelationId, request != null && request.getRecords() != null ? request.getRecords().size() : 0);

        try {
            validateForecastRequest(request);

            // Access allowlisted ForecastTool through ToolRegistry
            ToolContract<ForecastToolRequest, ForecastToolResponse> forecastTool =
                    toolRegistry.getAllowlistedTool(ForecastTool.TOOL_NAME);

            ForecastToolRequest toolRequest = ForecastToolRequest.builder()
                    .records(request.getRecords())
                    .correlationId(effectiveCorrelationId)
                    .build();

            ForecastToolResponse toolResponse = forecastTool.execute(toolRequest);

            long elapsed = System.currentTimeMillis() - startTime;
            metrics.recordForecastSuccess(elapsed);

            return ForecastResponseDto.builder()
                    .status(toolResponse.getStatus())
                    .modelVersion(toolResponse.getModelVersion())
                    .forecastCount(toolResponse.getForecastCount())
                    .forecasts(toolResponse.getForecasts())
                    .modelMetadata(toolResponse.getModelMetadata())
                    .provenance(toolResponse.getProvenance())
                    .build();

        } catch (Exception e) {
            metrics.recordForecastFailure(e.getClass().getSimpleName());
            throw e;
        }
    }

    @Override
    public HealthResponseDto getHealth(String correlationId) {
        if (correlationId != null && !correlationId.isBlank()) {
            provenanceTracker.setCurrentCorrelationId(correlationId);
        }
        FastApiHealthResponse downstreamHealth = inferenceClient.checkHealth();

        return HealthResponseDto.builder()
                .status("HEALTHY")
                .orchestrationLayer("UP")
                .downstreamService(downstreamHealth.getService())
                .downstreamModelLoaded(downstreamHealth.getModelLoaded())
                .timestampUtc(Instant.now().toString())
                .correlationId(provenanceTracker.getCurrentCorrelationId())
                .build();
    }

    @Override
    public ReadinessResponseDto getReadiness(String correlationId) {
        if (correlationId != null && !correlationId.isBlank()) {
            provenanceTracker.setCurrentCorrelationId(correlationId);
        }
        FastApiReadinessResponse downstreamReady = inferenceClient.checkReadiness();

        return ReadinessResponseDto.builder()
                .status(downstreamReady.getStatus())
                .modelVersion(downstreamReady.getModelVersion())
                .featureCount(downstreamReady.getFeatureCount())
                .scalerReady(downstreamReady.getScalerReady())
                .calibrationReady(downstreamReady.getCalibrationReady())
                .timestampUtc(Instant.now().toString())
                .correlationId(provenanceTracker.getCurrentCorrelationId())
                .build();
    }

    @Override
    public VersionResponseDto getModelVersion(String correlationId) {
        if (correlationId != null && !correlationId.isBlank()) {
            provenanceTracker.setCurrentCorrelationId(correlationId);
        }
        FastApiVersionResponse downstreamVersion = inferenceClient.getVersion();
        boolean verified = properties.getFastApi().getExpectedModelId().equals(downstreamVersion.getModelId());

        return VersionResponseDto.builder()
                .modelId(downstreamVersion.getModelId())
                .candidateId(downstreamVersion.getCandidateId())
                .architecture(downstreamVersion.getArchitecture())
                .parameterCount(downstreamVersion.getParameters())
                .modelSha256(properties.getFastApi().getExpectedModelSha256())
                .releaseStatus(downstreamVersion.getReleaseStatus())
                .releaseVersion("v1.0.0")
                .verifiedProductionIdentity(verified)
                .build();
    }

    private void validateForecastRequest(ForecastRequestDto request) {
        if (request == null || request.getRecords() == null || request.getRecords().isEmpty()) {
            throw new InvalidForecastRequestException("Forecast request records cannot be empty.");
        }
        if (request.getRecords().size() < FeatureRecordDto.REQUIRED_WINDOW_SIZE) {
            throw new InvalidForecastRequestException(
                    String.format("Insufficient sequence length: required at least %d rows (W=14), got %d",
                            FeatureRecordDto.REQUIRED_WINDOW_SIZE, request.getRecords().size()));
        }
    }
}
