package com.atmosiq.observability;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Basic observability metrics and execution telemetry counters.
 */
@Slf4j
@Getter
@Component
public class OrchestrationMetrics {

    private final AtomicLong forecastRequestsTotal = new AtomicLong(0);
    private final AtomicLong forecastSuccessTotal = new AtomicLong(0);
    private final AtomicLong forecastFailureTotal = new AtomicLong(0);
    private final AtomicLong lastExecutionLatencyMs = new AtomicLong(0);

    public void recordForecastRequest() {
        forecastRequestsTotal.incrementAndGet();
    }

    public void recordForecastSuccess(long latencyMs) {
        forecastSuccessTotal.incrementAndGet();
        lastExecutionLatencyMs.set(latencyMs);
        log.info("Forecast executed successfully in {} ms [Total Success: {}]", latencyMs, forecastSuccessTotal.get());
    }

    public void recordForecastFailure(String errorType) {
        forecastFailureTotal.incrementAndGet();
        log.warn("Forecast execution failed with error: '{}' [Total Failures: {}]", errorType, forecastFailureTotal.get());
    }
}
