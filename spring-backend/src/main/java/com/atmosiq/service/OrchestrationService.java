package com.atmosiq.service;

import com.atmosiq.model.request.ForecastRequestDto;
import com.atmosiq.model.response.ForecastResponseDto;
import com.atmosiq.model.response.HealthResponseDto;
import com.atmosiq.model.response.ReadinessResponseDto;
import com.atmosiq.model.response.VersionResponseDto;

/**
 * Core orchestration service for AtmosIQ. Coordinates tools, validation, and downstream inference.
 */
public interface OrchestrationService {

    ForecastResponseDto executeForecast(ForecastRequestDto request, String correlationId);

    HealthResponseDto getHealth(String correlationId);

    ReadinessResponseDto getReadiness(String correlationId);

    VersionResponseDto getModelVersion(String correlationId);
}
