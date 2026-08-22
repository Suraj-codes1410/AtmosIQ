package com.atmosiq.tools;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.InvalidForecastRequestException;
import com.atmosiq.model.fastapi.FastApiForecastItem;
import com.atmosiq.model.fastapi.FastApiPredictRequest;
import com.atmosiq.model.fastapi.FastApiPredictResponse;
import com.atmosiq.model.request.FeatureRecordDto;
import com.atmosiq.model.response.ForecastItemDto;
import com.atmosiq.provenance.ProvenanceTracker;
import com.atmosiq.test.TestFixtures;
import com.atmosiq.tools.forecast.ForecastTool;
import com.atmosiq.tools.forecast.ForecastToolRequest;
import com.atmosiq.tools.forecast.ForecastToolResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ForecastToolTest {

    @Mock
    private FastApiInferenceClient inferenceClient;

    private AtmosIQProperties properties;
    private ProvenanceTracker provenanceTracker;
    private ForecastTool forecastTool;

    @BeforeEach
    void setUp() {
        properties = new AtmosIQProperties();
        provenanceTracker = new ProvenanceTracker();
        forecastTool = new ForecastTool(inferenceClient, properties, provenanceTracker);
    }

    @Test
    @DisplayName("ForecastTool name and description are correctly defined")
    void testToolMetadata() {
        assertThat(forecastTool.getName()).isEqualTo("forecast_pm25");
        assertThat(forecastTool.getDescription()).contains("AtmosIQ TCN v1.0.0");
    }

    @Test
    @DisplayName("ForecastTool executes valid request, maps uncertainty and provenance")
    void testExecute_Success() {
        FastApiForecastItem item = FastApiForecastItem.builder()
                .predictionId("pred_test_123")
                .timestampUtc("2024-01-14")
                .forecastPm25(120.5)
                .lower90(24.84)
                .upper90(216.16)
                .conformalHalfWidth(95.66)
                .build();

        FastApiPredictResponse mockResponse = FastApiPredictResponse.builder()
                .status("SUCCESS")
                .modelVersion("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .executionLatencyMs(2.5)
                .batchSize(1)
                .forecasts(List.of(item))
                .build();

        when(inferenceClient.predict(any(FastApiPredictRequest.class))).thenReturn(mockResponse);

        ForecastToolRequest toolRequest = ForecastToolRequest.builder()
                .records(TestFixtures.createValid14DaySequence())
                .correlationId("corr_unit_test")
                .build();

        ForecastToolResponse response = forecastTool.execute(toolRequest);

        assertThat(response).isNotNull();
        assertThat(response.getStatus()).isEqualTo("SUCCESS");
        assertThat(response.getModelVersion()).isEqualTo("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0");
        assertThat(response.getForecastCount()).isEqualTo(1);

        ForecastItemDto forecast = response.getForecasts().get(0);
        assertThat(forecast.getForecastPm25()).isEqualTo(120.5);
        assertThat(forecast.getUncertainty().getLowerBound()).isEqualTo(24.84);
        assertThat(forecast.getUncertainty().getUpperBound()).isEqualTo(216.16);
        assertThat(forecast.getUncertainty().getConformalHalfWidth()).isEqualTo(95.66);
        assertThat(forecast.getUncertainty().getConfidenceLevel()).isEqualTo(0.90);
        assertThat(forecast.getUncertainty().getPhysicalDisclaimer())
                .isEqualTo("PREDICTION INTERVAL != GUARANTEED PHYSICAL UNCERTAINTY");

        assertThat(response.getProvenance()).isNotNull();
        assertThat(response.getProvenance().getCorrelationId()).isEqualTo("corr_unit_test");
        assertThat(response.getProvenance().getToolName()).isEqualTo("forecast_pm25");

        assertThat(response.getModelMetadata()).isNotNull();
        assertThat(response.getModelMetadata().getParameterCount()).isEqualTo(849);
        assertThat(response.getModelMetadata().getSequenceWindow()).isEqualTo(14);
        assertThat(response.getModelMetadata().getFeatureDimension()).isEqualTo(35);

        verify(inferenceClient).predict(any(FastApiPredictRequest.class));
    }

    @Test
    @DisplayName("ForecastTool rejects sequence shorter than 14 rows (W=14)")
    void testExecute_SequenceTooShort_ThrowsException() {
        List<Map<String, Object>> shortList = TestFixtures.createValid14DaySequence().subList(0, 10);

        ForecastToolRequest toolRequest = ForecastToolRequest.builder()
                .records(shortList)
                .build();

        assertThatThrownBy(() -> forecastTool.execute(toolRequest))
                .isInstanceOf(InvalidForecastRequestException.class)
                .hasMessageContaining("Insufficient sequence length: required at least 14 rows");
    }

    @Test
    @DisplayName("ForecastTool rejects sequence missing required 35 features")
    void testExecute_MissingFeatures_ThrowsException() {
        List<Map<String, Object>> records = new ArrayList<>();
        for (int i = 0; i < 14; i++) {
            Map<String, Object> row = new HashMap<>();
            row.put("date", "2024-01-01");
            row.put("pm25_lag_1d", 100.0); // missing 34 features
            records.add(row);
        }

        ForecastToolRequest toolRequest = ForecastToolRequest.builder()
                .records(records)
                .build();

        assertThatThrownBy(() -> forecastTool.execute(toolRequest))
                .isInstanceOf(InvalidForecastRequestException.class)
                .hasMessageContaining("Missing required prediction-safe features");
    }
}
