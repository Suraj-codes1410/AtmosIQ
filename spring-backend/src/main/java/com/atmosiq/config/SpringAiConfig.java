package com.atmosiq.config;

import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.model.response.ModelMetadataDto;
import com.atmosiq.tools.forecast.ForecastTool;
import com.atmosiq.tools.forecast.ForecastToolRequest;
import com.atmosiq.tools.forecast.ForecastToolResponse;
import com.atmosiq.tools.health.HealthTool;
import com.atmosiq.tools.metadata.ModelMetadataTool;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Description;

import java.util.Map;
import java.util.function.Function;

/**
 * Spring AI Tool Registration & Schema definitions.
 * Exposes allowlisted tools as callable functions with strict descriptions and parameter validation.
 */
@Configuration
public class SpringAiConfig {

    @Bean
    @Description("Obtain a PM2.5 forecast from the certified AtmosIQ production forecasting service. " +
            "Requires at least 14 daily rows containing the 35 prediction-safe environmental features.")
    public Function<ForecastToolRequest, ForecastToolResponse> forecastPm25(ForecastTool forecastTool) {
        return forecastTool;
    }

    @Bean
    @Description("Check the operational health and readiness of the certified AtmosIQ inference service.")
    public Function<Map<String, Object>, FastApiHealthResponse> checkServiceHealth(HealthTool healthTool) {
        return healthTool;
    }

    @Bean
    @Description("Retrieve the architectural parameters, parameter count (849), and cryptographic release identity for AtmosIQ v1.0.0.")
    public Function<Map<String, Object>, ModelMetadataDto> getModelMetadata(ModelMetadataTool metadataTool) {
        return metadataTool;
    }
}
