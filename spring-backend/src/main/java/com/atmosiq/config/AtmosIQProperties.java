package com.atmosiq.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Externalized configuration properties for AtmosIQ orchestration layer.
 */
@Data
@Component
@ConfigurationProperties(prefix = "atmosiq")
public class AtmosIQProperties {

    private FastApiProperties fastApi = new FastApiProperties();
    private OrchestrationProperties orchestration = new OrchestrationProperties();

    @Data
    public static class FastApiProperties {
        private String baseUrl = "http://localhost:8000";
        private int connectTimeoutMs = 5000;
        private int readTimeoutMs = 10000;
        private String expectedModelId = "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0";
        private String expectedModelSha256 = "fdc99f7ca4410f3d577e52718e35c956c97f368cd91a8b7f505ee23824085bac";
        private boolean failOnModelIdMismatch = true;
    }

    @Data
    public static class OrchestrationProperties {
        private String defaultHorizon = "24h";
        private List<String> allowlistedTools = List.of("forecast_pm25", "get_model_metadata", "check_service_health");
        private boolean enforceToolAllowlist = true;
    }
}
