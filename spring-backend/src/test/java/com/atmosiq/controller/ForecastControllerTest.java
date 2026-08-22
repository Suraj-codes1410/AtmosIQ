package com.atmosiq.controller;

import com.atmosiq.exception.FastApiUnavailableException;
import com.atmosiq.exception.InvalidForecastRequestException;
import com.atmosiq.model.request.ForecastRequestDto;
import com.atmosiq.model.response.ForecastItemDto;
import com.atmosiq.model.response.ForecastResponseDto;
import com.atmosiq.model.response.ModelMetadataDto;
import com.atmosiq.model.response.ProvenanceMetadataDto;
import com.atmosiq.model.response.UncertaintyDto;
import com.atmosiq.provenance.ProvenanceTracker;
import com.atmosiq.service.OrchestrationService;
import com.atmosiq.test.TestFixtures;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ExtendWith(MockitoExtension.class)
class ForecastControllerTest {

    @Mock
    private OrchestrationService orchestrationService;

    private MockMvc mockMvc;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @BeforeEach
    void setUp() {
        ProvenanceTracker provenanceTracker = new ProvenanceTracker();
        ForecastController controller = new ForecastController(orchestrationService);
        GlobalExceptionHandler exceptionHandler = new GlobalExceptionHandler(provenanceTracker);

        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(exceptionHandler)
                .build();
    }

    @Test
    @DisplayName("POST /api/v1/forecast returns 200 OK with forecast and uncertainty")
    void testPostForecast_Success() throws Exception {
        ForecastResponseDto mockResponse = ForecastResponseDto.builder()
                .status("SUCCESS")
                .modelVersion("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .forecastCount(1)
                .forecasts(List.of(ForecastItemDto.builder()
                        .predictionId("pred_mvc_123")
                        .timestampUtc("2024-01-14")
                        .forecastPm25(108.5)
                        .uncertainty(UncertaintyDto.builder()
                                .lowerBound(12.84)
                                .upperBound(204.16)
                                .conformalHalfWidth(95.66)
                                .confidenceLevel(0.90)
                                .build())
                        .build()))
                .modelMetadata(ModelMetadataDto.builder()
                        .modelId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                        .architecture("TCN")
                        .parameterCount(849)
                        .build())
                .provenance(ProvenanceMetadataDto.builder()
                        .correlationId("corr_mvc_test")
                        .build())
                .build();

        when(orchestrationService.executeForecast(any(ForecastRequestDto.class), any())).thenReturn(mockResponse);

        ForecastRequestDto request = TestFixtures.createValidForecastRequest();

        mockMvc.perform(post("/api/v1/forecast")
                        .contentType(MediaType.APPLICATION_JSON)
                        .header("X-Correlation-ID", "corr_mvc_test")
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("SUCCESS"))
                .andExpect(jsonPath("$.modelVersion").value("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0"))
                .andExpect(jsonPath("$.forecasts[0].forecastPm25").value(108.5))
                .andExpect(jsonPath("$.forecasts[0].uncertainty.conformalHalfWidth").value(95.66));
    }

    @Test
    @DisplayName("POST /api/v1/forecast returns 503 SERVICE_UNAVAILABLE when downstream is down")
    void testPostForecast_DownstreamUnavailable_Returns503() throws Exception {
        when(orchestrationService.executeForecast(any(ForecastRequestDto.class), any()))
                .thenThrow(new FastApiUnavailableException("Downstream FastAPI unreachable"));

        ForecastRequestDto request = TestFixtures.createValidForecastRequest();

        mockMvc.perform(post("/api/v1/forecast")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error").value("DOWNSTREAM_SERVICE_UNAVAILABLE"))
                .andExpect(jsonPath("$.message").value("Downstream FastAPI unreachable"));
    }

    @Test
    @DisplayName("GET /api/v1/forecast/features returns 35 certified prediction features")
    void testGetFeatures_Returns35Features() throws Exception {
        mockMvc.perform(get("/api/v1/forecast/features"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.featureCount").value(35))
                .andExpect(jsonPath("$.requiredWindowSize").value(14))
                .andExpect(jsonPath("$.features").isArray());
    }
}
