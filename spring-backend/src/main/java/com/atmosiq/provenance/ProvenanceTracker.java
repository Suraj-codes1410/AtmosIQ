package com.atmosiq.provenance;

import org.springframework.stereotype.Component;

import java.util.UUID;

/**
 * Thread-safe provenance and correlation ID tracking component.
 */
@Component
public class ProvenanceTracker {

    private static final ThreadLocal<String> CORRELATION_ID_HOLDER = new ThreadLocal<>();

    public void setCurrentCorrelationId(String correlationId) {
        if (correlationId != null && !correlationId.isBlank()) {
            CORRELATION_ID_HOLDER.set(correlationId);
        } else {
            CORRELATION_ID_HOLDER.set(generateCorrelationId());
        }
    }

    public String getCurrentCorrelationId() {
        String id = CORRELATION_ID_HOLDER.get();
        if (id == null || id.isBlank()) {
            id = generateCorrelationId();
            CORRELATION_ID_HOLDER.set(id);
        }
        return id;
    }

    public void clear() {
        CORRELATION_ID_HOLDER.remove();
    }

    public String generateRequestId() {
        return "req_" + UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }

    public String generateCorrelationId() {
        return "corr_" + UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }
}
