package com.atmosiq.controller;

import com.atmosiq.model.response.HealthResponseDto;
import com.atmosiq.model.response.ReadinessResponseDto;
import com.atmosiq.model.response.VersionResponseDto;
import com.atmosiq.provenance.ProvenanceTracker;
import com.atmosiq.service.OrchestrationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ExtendWith(MockitoExtension.class)
class MonitoringControllerTest {

    @Mock
    private OrchestrationService orchestrationService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        ProvenanceTracker provenanceTracker = new ProvenanceTracker();
        MonitoringController controller = new MonitoringController(orchestrationService);
        GlobalExceptionHandler exceptionHandler = new GlobalExceptionHandler(provenanceTracker);

        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(exceptionHandler)
                .build();
    }

    @Test
    @DisplayName("GET /api/v1/health returns orchestration and downstream status")
    void testGetHealth() throws Exception {
        HealthResponseDto mockHealth = HealthResponseDto.builder()
                .status("HEALTHY")
                .orchestrationLayer("UP")
                .downstreamService("AtmosIQ_Production_Service")
                .downstreamModelLoaded(true)
                .build();

        when(orchestrationService.getHealth(any())).thenReturn(mockHealth);

        mockMvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("HEALTHY"))
                .andExpect(jsonPath("$.orchestrationLayer").value("UP"))
                .andExpect(jsonPath("$.downstreamModelLoaded").value(true));
    }

    @Test
    @DisplayName("GET /api/v1/ready returns readiness of scaler and calibration")
    void testGetReadiness() throws Exception {
        ReadinessResponseDto mockReady = ReadinessResponseDto.builder()
                .status("READY")
                .modelVersion("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .featureCount(35)
                .scalerReady(true)
                .calibrationReady(true)
                .build();

        when(orchestrationService.getReadiness(any())).thenReturn(mockReady);

        mockMvc.perform(get("/api/v1/ready"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("READY"))
                .andExpect(jsonPath("$.featureCount").value(35))
                .andExpect(jsonPath("$.calibrationReady").value(true));
    }

    @Test
    @DisplayName("GET /api/v1/model returns verified production architecture & parameters")
    void testGetModel() throws Exception {
        VersionResponseDto mockVersion = VersionResponseDto.builder()
                .modelId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0")
                .architecture("TCN")
                .parameterCount(849)
                .releaseVersion("v1.0.0")
                .verifiedProductionIdentity(true)
                .build();

        when(orchestrationService.getModelVersion(any())).thenReturn(mockVersion);

        mockMvc.perform(get("/api/v1/model"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.modelId").value("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0"))
                .andExpect(jsonPath("$.architecture").value("TCN"))
                .andExpect(jsonPath("$.parameterCount").value(849))
                .andExpect(jsonPath("$.verifiedProductionIdentity").value(true));
    }
}
