package com.atmosiq.service;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.InvalidForecastRequestException;
import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.model.fastapi.FastApiReadinessResponse;
import com.atmosiq.model.fastapi.FastApiVersionResponse;
import com.atmosiq.model.request.ForecastRequestDto;
import com.atmosiq.model.response.ForecastItemDto;
import com.atmosiq.model.response.ForecastResponseDto;
import com.atmosiq.model.response.HealthResponseDto;
import com.atmosiq.model.response.ModelMetadataDto;
import com.atmosiq.model.response.ProvenanceMetadataDto;
import com.atmosiq.model.response.ReadinessResponseDto;
import com.atmosiq.model.response.UncertaintyDto;
import com.atmosiq.model.response.VersionResponseDto;
import com.atmosiq.observability.OrchestrationMetrics;
import com.atmosiq.provenance.ProvenanceTracker;
import com.atmosiq.test.TestFixtures;
import com.atmosiq.tools.ToolRegistry;
import com.atmosiq.tools.forecast.ForecastTool;
import com.atmosiq.tools.forecast.ForecastToolRequest;
import com.atmosiq.tools.forecast.ForecastToolResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class OrchestrationServiceTest {

    @Mock
    private ToolRegistry toolRegistry;

    @Mock
    private ForecastTool forecastTool;

    @Mock
    private FastApiInferenceClient inferenceClient;

    private AtmosIQProperties properties;
    private ProvenanceTracker provenanceTracker;
    private OrchestrationMetrics metrics;
    private OrchestrationService orchestrationService;

    @BeforeEach
    void setUp() {
        properties = new AtmosIQProperties();
        provenanceTracker = new ProvenanceTracker();
        metrics = new OrchestrationMetrics();

        orchestrationService = new OrchestrationServiceImpl(
                toolRegistry,
                inferenceClient,
                properties,
                provenanceTracker,
                metrics
        );
    }

    @Test
    @DisplayName("OrchestrationService executes forecast successfully through tool registry")
    void testExecuteForecast_Success() {
        ForecastToolResponse mockToolResp = ForecastToolResponse.builder()
                .status("SUCCESS")
                .modelVersion("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .forecastCount(1)
                .forecasts(List.of(ForecastItemDto.builder()
                        .predictionId("p123")
                        .forecastPm25(95.4)
                        .uncertainty(UncertaintyDto.builder()
                                .lowerBound(0.0)
                                .upperBound(191.06)
                                .confidenceLevel(0.90)
                                .build())
                        .build()))
                .modelMetadata(ModelMetadataDto.builder().modelId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0").build())
                .provenance(ProvenanceMetadataDto.builder().correlationId("corr_orch_test").build())
                .build();

        org.mockito.Mockito.doReturn(forecastTool).when(toolRegistry).getAllowlistedTool("forecast_pm25");
        when(forecastTool.execute(any(ForecastToolRequest.class))).thenReturn(mockToolResp);

        ForecastRequestDto request = TestFixtures.createValidForecastRequest();
        ForecastResponseDto response = orchestrationService.executeForecast(request, "corr_orch_test");

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo("SUCCESS");
        assertThat(response.getForecastCount()).isEqualTo(1);
        assertThat(response.getForecasts().get(0).getForecastPm25()).isEqualTo(95.4);
        assertThat(metrics.getForecastSuccessTotal().get()).isEqualTo(1);

        verify(toolRegistry).getAllowlistedTool("forecast_pm25");
        verify(forecastTool).execute(any(ForecastToolRequest.class));
    }

    @Test
    @DisplayName("OrchestrationService rejects null records with InvalidForecastRequestException")
    void testExecuteForecast_NullRecords_ThrowsException() {
        ForecastRequestDto request = ForecastRequestDto.builder().records(null).build();

        assertThatThrownBy(() -> orchestrationService.executeForecast(request, "corr_test"))
                .isInstanceOf(InvalidForecastRequestException.class);

        assertThat(metrics.getForecastFailureTotal().get()).isEqualTo(1);
    }

    @Test
    @DisplayName("OrchestrationService gets health successfully")
    void testGetHealth_Success() {
        FastApiHealthResponse mockHealth = FastApiHealthResponse.builder()
                .status("HEALTHY")
                .service("AtmosIQ_Production_Service")
                .modelLoaded(true)
                .build();

        when(inferenceClient.checkHealth()).thenReturn(mockHealth);

        HealthResponseDto resp = orchestrationService.getHealth("corr_health_test");

        assertThat(resp).isNotNull();
        assertThat(resp.getStatus()).isEqualTo("HEALTHY");
        assertThat(resp.getDownstreamModelLoaded()).isTrue();
        assertThat(resp.getDownstreamService()).isEqualTo("AtmosIQ_Production_Service");
    }

    @Test
    @DisplayName("OrchestrationService gets readiness successfully")
    void testGetReadiness_Success() {
        FastApiReadinessResponse mockReady = FastApiReadinessResponse.builder()
                .status("READY")
                .modelVersion("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .featureCount(35)
                .scalerReady(true)
                .calibrationReady(true)
                .build();

        when(inferenceClient.checkReadiness()).thenReturn(mockReady);

        ReadinessResponseDto resp = orchestrationService.getReadiness("corr_ready_test");

        assertThat(resp).isNotNull();
        assertThat(resp.getStatus()).isEqualTo("READY");
        assertThat(resp.getFeatureCount()).isEqualTo(35);
        assertThat(resp.getScalerReady()).isTrue();
        assertThat(resp.getCalibrationReady()).isTrue();
    }

    @Test
    @DisplayName("OrchestrationService gets model version and verifies identity")
    void testGetModelVersion_Success() {
        FastApiVersionResponse mockVer = FastApiVersionResponse.builder()
                .modelId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .candidateId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0")
                .architecture("TCN")
                .parameters(849)
                .releaseStatus("RELEASE_CERTIFIED")
                .build();

        when(inferenceClient.getVersion()).thenReturn(mockVer);

        VersionResponseDto resp = orchestrationService.getModelVersion("corr_ver_test");

        assertThat(resp).isNotNull();
        assertThat(resp.getModelId()).isEqualTo("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0");
        assertThat(resp.getVerifiedProductionIdentity()).isTrue();
        assertThat(resp.getReleaseVersion()).isEqualTo("v1.0.0");
    }
}
