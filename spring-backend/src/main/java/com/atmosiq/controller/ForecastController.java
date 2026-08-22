package com.atmosiq.controller;

import com.atmosiq.model.request.FeatureRecordDto;
import com.atmosiq.model.request.ForecastRequestDto;
import com.atmosiq.model.response.ForecastResponseDto;
import com.atmosiq.observability.CorrelationIdFilter;
import com.atmosiq.service.OrchestrationService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/**
 * REST API for executing certified PM2.5 forecasting requests through AtmosIQ orchestration.
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/forecast")
@RequiredArgsConstructor
public class ForecastController {

    private final OrchestrationService orchestrationService;

    @PostMapping
    public ResponseEntity<ForecastResponseDto> createForecast(
            @Valid @RequestBody ForecastRequestDto request,
            @RequestHeader(value = CorrelationIdFilter.CORRELATION_ID_HEADER, required = false) String correlationId
    ) {
        log.info("Received forecast request [CorrelationId: {}]", correlationId);
        ForecastResponseDto response = orchestrationService.executeForecast(request, correlationId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/features")
    public ResponseEntity<Map<String, Object>> getCertifiedFeatures() {
        return ResponseEntity.ok(Map.of(
                "featureCount", FeatureRecordDto.REQUIRED_FEATURE_DIM,
                "requiredWindowSize", FeatureRecordDto.REQUIRED_WINDOW_SIZE,
                "features", FeatureRecordDto.CERTIFIED_35_FEATURES
        ));
    }
}
