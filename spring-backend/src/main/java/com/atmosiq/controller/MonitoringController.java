package com.atmosiq.controller;

import com.atmosiq.model.response.HealthResponseDto;
import com.atmosiq.model.response.ReadinessResponseDto;
import com.atmosiq.model.response.VersionResponseDto;
import com.atmosiq.observability.CorrelationIdFilter;
import com.atmosiq.service.OrchestrationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Monitoring and health introspection endpoints reflecting the orchestration and downstream service state.
 */
@Slf4j
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class MonitoringController {

    private final OrchestrationService orchestrationService;

    @GetMapping("/health")
    public ResponseEntity<HealthResponseDto> getHealth(
            @RequestHeader(value = CorrelationIdFilter.CORRELATION_ID_HEADER, required = false) String correlationId
    ) {
        return ResponseEntity.ok(orchestrationService.getHealth(correlationId));
    }

    @GetMapping("/ready")
    public ResponseEntity<ReadinessResponseDto> getReadiness(
            @RequestHeader(value = CorrelationIdFilter.CORRELATION_ID_HEADER, required = false) String correlationId
    ) {
        return ResponseEntity.ok(orchestrationService.getReadiness(correlationId));
    }

    @GetMapping("/model")
    public ResponseEntity<VersionResponseDto> getModelVersion(
            @RequestHeader(value = CorrelationIdFilter.CORRELATION_ID_HEADER, required = false) String correlationId
    ) {
        return ResponseEntity.ok(orchestrationService.getModelVersion(correlationId));
    }
}
