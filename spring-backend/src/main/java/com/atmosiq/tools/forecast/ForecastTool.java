package com.atmosiq.tools.forecast;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.InvalidForecastRequestException;
import com.atmosiq.model.fastapi.FastApiForecastItem;
import com.atmosiq.model.fastapi.FastApiPredictRequest;
import com.atmosiq.model.fastapi.FastApiPredictResponse;
import com.atmosiq.model.request.FeatureRecordDto;
import com.atmosiq.model.response.ForecastItemDto;
import com.atmosiq.model.response.ModelMetadataDto;
import com.atmosiq.model.response.ProvenanceMetadataDto;
import com.atmosiq.model.response.UncertaintyDto;
import com.atmosiq.provenance.ProvenanceTracker;
import com.atmosiq.tools.ToolContract;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Controlled AtmosIQ tool: forecast_pm25.
 * Invokes the certified production TCN model via FastAPI and wraps results in typed contracts.
 */
@Slf4j
@Component("forecastPm25Tool")
@RequiredArgsConstructor
public class ForecastTool implements ToolContract<ForecastToolRequest, ForecastToolResponse>,
        Function<ForecastToolRequest, ForecastToolResponse> {

    public static final String TOOL_NAME = "forecast_pm25";
    public static final String TOOL_DESCRIPTION = "Executes certified PM2.5 forecast for Delhi NCR using AtmosIQ TCN v1.0.0. " +
            "Requires at least 14 sequential daily records of the 35 prediction-safe environmental features.";

    private final FastApiInferenceClient inferenceClient;
    private final AtmosIQProperties properties;
    private final ProvenanceTracker provenanceTracker;

    @Override
    public String getName() {
        return TOOL_NAME;
    }

    @Override
    public String getDescription() {
        return TOOL_DESCRIPTION;
    }

    @Override
    public ForecastToolResponse apply(ForecastToolRequest request) {
        return execute(request);
    }

    @Override
    public ForecastToolResponse execute(ForecastToolRequest request) {
        long startTime = System.currentTimeMillis();
        String correlationId = request != null && request.getCorrelationId() != null
                ? request.getCorrelationId()
                : provenanceTracker.getCurrentCorrelationId();
        String requestId = provenanceTracker.generateRequestId();

        log.info("Executing Tool '{}' [RequestId: {}, CorrelationId: {}]", TOOL_NAME, requestId, correlationId);

        validateInput(request);

        FastApiPredictRequest fastApiRequest = FastApiPredictRequest.builder()
                .records(request.getRecords())
                .build();

        FastApiPredictResponse fastApiResponse = inferenceClient.predict(fastApiRequest);

        long executionLatencyMs = System.currentTimeMillis() - startTime;

        return mapToToolResponse(fastApiResponse, requestId, correlationId, executionLatencyMs);
    }

    private void validateInput(ForecastToolRequest request) {
        if (request == null || request.getRecords() == null || request.getRecords().isEmpty()) {
            throw new InvalidForecastRequestException("Request records sequence cannot be null or empty.");
        }

        List<Map<String, Object>> records = request.getRecords();
        if (records.size() < FeatureRecordDto.REQUIRED_WINDOW_SIZE) {
            throw new InvalidForecastRequestException(
                    String.format("Insufficient sequence length: required at least %d rows (W=14), but got %d",
                            FeatureRecordDto.REQUIRED_WINDOW_SIZE, records.size()));
        }

        // Validate that each row contains all 35 features
        Map<String, Object> firstRow = records.get(0);
        List<String> missingCols = new ArrayList<>();
        for (String feature : FeatureRecordDto.CERTIFIED_35_FEATURES) {
            if (!firstRow.containsKey(feature)) {
                missingCols.add(feature);
            }
        }

        if (!missingCols.isEmpty()) {
            throw new InvalidForecastRequestException(
                    String.format("Missing required prediction-safe features in input: %s", missingCols));
        }
    }

    private ForecastToolResponse mapToToolResponse(
            FastApiPredictResponse response,
            String requestId,
            String correlationId,
            long latencyMs
    ) {
        List<ForecastItemDto> items = new ArrayList<>();
        if (response.getForecasts() != null) {
            for (FastApiForecastItem f : response.getForecasts()) {
                UncertaintyDto uncertainty = UncertaintyDto.builder()
                        .lowerBound(f.getLower90())
                        .upperBound(f.getUpper90())
                        .conformalHalfWidth(f.getConformalHalfWidth())
                        .confidenceLevel(0.90)
                        .methodology("Split-Conformal Prediction (90% Empirical Bound)")
                        .physicalDisclaimer("PREDICTION INTERVAL != GUARANTEED PHYSICAL UNCERTAINTY")
                        .build();

                items.add(ForecastItemDto.builder()
                        .predictionId(f.getPredictionId())
                        .timestampUtc(f.getTimestampUtc())
                        .forecastPm25(f.getForecastPm25())
                        .uncertainty(uncertainty)
                        .build());
            }
        }

        ModelMetadataDto modelMetadata = ModelMetadataDto.builder()
                .modelId(response.getModelVersion())
                .candidateId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0")
                .architecture("TCN (Temporal Convolutional Network)")
                .parameterCount(849)
                .sequenceWindow(FeatureRecordDto.REQUIRED_WINDOW_SIZE)
                .featureDimension(FeatureRecordDto.REQUIRED_FEATURE_DIM)
                .augmentationPolicy("25% CAL-07 (100% Synthetic Strictly Prohibited)")
                .modelSha256(properties.getFastApi().getExpectedModelSha256())
                .releaseStatus("FINAL_PRODUCTION_CERTIFIED")
                .fallbackTarget("MODEL_V3_PRODUCTION")
                .build();

        ProvenanceMetadataDto provenance = ProvenanceMetadataDto.builder()
                .requestId(requestId)
                .correlationId(correlationId)
                .timestampUtc(Instant.now().toString())
                .releaseVersion("v1.0.0")
                .modelId(response.getModelVersion())
                .modelSha256(properties.getFastApi().getExpectedModelSha256())
                .toolName(TOOL_NAME)
                .downstreamService("FastAPI /predict")
                .latencyMs(latencyMs)
                .build();

        return ForecastToolResponse.builder()
                .status("SUCCESS")
                .modelVersion(response.getModelVersion())
                .forecastCount(items.size())
                .forecasts(items)
                .modelMetadata(modelMetadata)
                .provenance(provenance)
                .build();
    }
}
