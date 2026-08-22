package com.atmosiq.ai;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.config.SpringAiConfig;
import com.atmosiq.model.fastapi.FastApiForecastItem;
import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.model.fastapi.FastApiPredictRequest;
import com.atmosiq.model.fastapi.FastApiPredictResponse;
import com.atmosiq.model.fastapi.FastApiVersionResponse;
import com.atmosiq.model.response.ModelMetadataDto;
import com.atmosiq.provenance.ProvenanceTracker;
import com.atmosiq.test.TestFixtures;
import com.atmosiq.tools.forecast.ForecastTool;
import com.atmosiq.tools.forecast.ForecastToolRequest;
import com.atmosiq.tools.forecast.ForecastToolResponse;
import com.atmosiq.tools.health.HealthTool;
import com.atmosiq.tools.metadata.ModelMetadataTool;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class SpringAiToolIntegrationTest {

    @Mock
    private FastApiInferenceClient inferenceClient;

    private SpringAiConfig springAiConfig;
    private ForecastTool forecastTool;
    private HealthTool healthTool;
    private ModelMetadataTool metadataTool;

    @BeforeEach
    void setUp() {
        AtmosIQProperties properties = new AtmosIQProperties();
        ProvenanceTracker provenanceTracker = new ProvenanceTracker();

        forecastTool = new ForecastTool(inferenceClient, properties, provenanceTracker);
        healthTool = new HealthTool(inferenceClient);
        metadataTool = new ModelMetadataTool(inferenceClient, properties);

        springAiConfig = new SpringAiConfig();
    }

    @Test
    @DisplayName("Spring AI forecast_pm25 functional tool executes cleanly")
    void testForecastPm25Tool_FunctionExecution() {
        Function<ForecastToolRequest, ForecastToolResponse> function = springAiConfig.forecastPm25(forecastTool);

        FastApiPredictResponse mockResp = FastApiPredictResponse.builder()
                .status("SUCCESS")
                .modelVersion("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .executionLatencyMs(2.1)
                .batchSize(1)
                .forecasts(List.of(FastApiForecastItem.builder()
                        .predictionId("ai_tool_pred_1")
                        .forecastPm25(102.3)
                        .lower90(6.64)
                        .upper90(197.96)
                        .conformalHalfWidth(95.66)
                        .build()))
                .build();

        when(inferenceClient.predict(any(FastApiPredictRequest.class))).thenReturn(mockResp);

        ForecastToolRequest req = ForecastToolRequest.builder()
                .records(TestFixtures.createValid14DaySequence())
                .correlationId("corr_ai_test")
                .build();

        ForecastToolResponse resp = function.apply(req);

        assertThat(resp).isNotNull();
        assertThat(resp.getStatus()).isEqualTo("SUCCESS");
        assertThat(resp.getForecasts().get(0).getForecastPm25()).isEqualTo(102.3);
    }

    @Test
    @DisplayName("Spring AI checkServiceHealth functional tool executes cleanly")
    void testCheckServiceHealthTool_FunctionExecution() {
        Function<Map<String, Object>, FastApiHealthResponse> function = springAiConfig.checkServiceHealth(healthTool);

        when(inferenceClient.checkHealth()).thenReturn(FastApiHealthResponse.builder()
                .status("HEALTHY")
                .service("AtmosIQ_Production_Service")
                .modelLoaded(true)
                .build());

        FastApiHealthResponse resp = function.apply(Map.of());

        assertThat(resp).isNotNull();
        assertThat(resp.getStatus()).isEqualTo("HEALTHY");
    }

    @Test
    @DisplayName("Spring AI getModelMetadata functional tool executes cleanly")
    void testGetModelMetadataTool_FunctionExecution() {
        Function<Map<String, Object>, ModelMetadataDto> function = springAiConfig.getModelMetadata(metadataTool);

        when(inferenceClient.getVersion()).thenReturn(FastApiVersionResponse.builder()
                .modelId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .candidateId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0")
                .architecture("TCN")
                .parameters(849)
                .build());

        ModelMetadataDto resp = function.apply(Map.of());

        assertThat(resp).isNotNull();
        assertThat(resp.getModelId()).isEqualTo("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0");
        assertThat(resp.getParameterCount()).isEqualTo(849);
    }
}
