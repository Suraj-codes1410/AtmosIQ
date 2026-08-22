package com.atmosiq.client.fastapi;

import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.DownstreamContractViolationException;
import com.atmosiq.exception.FastApiTimeoutException;
import com.atmosiq.exception.FastApiUnavailableException;
import com.atmosiq.exception.InferenceContractException;
import com.atmosiq.exception.ModelIdentityMismatchException;
import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.model.fastapi.FastApiPredictRequest;
import com.atmosiq.model.fastapi.FastApiPredictResponse;
import com.atmosiq.model.fastapi.FastApiReadinessResponse;
import com.atmosiq.model.fastapi.FastApiVersionResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpTimeoutException;

/**
 * Production implementation of FastApiInferenceClient using Spring RestClient.
 * Enforces timeouts, strict response validation, and model identity immutability guards.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RestClientFastApiInferenceClient implements FastApiInferenceClient {

    private final RestClient fastApiRestClient;
    private final AtmosIQProperties properties;

    @Override
    public FastApiHealthResponse checkHealth() {
        log.debug("Executing GET /health against FastAPI inference service");
        try {
            FastApiHealthResponse response = fastApiRestClient.get()
                    .uri("/health")
                    .accept(MediaType.APPLICATION_JSON)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, (req, resp) -> {
                        throw new DownstreamContractViolationException(
                                "FastAPI health check failed with status: " + resp.getStatusCode());
                    })
                    .body(FastApiHealthResponse.class);

            if (response == null || !"HEALTHY".equalsIgnoreCase(response.getStatus())) {
                throw new DownstreamContractViolationException("FastAPI returned non-healthy status: " + response);
            }
            return response;
        } catch (Exception e) {
            handleClientException("checkHealth", e);
            throw new FastApiUnavailableException("Failed to check FastAPI health", e);
        }
    }

    @Override
    public FastApiReadinessResponse checkReadiness() {
        log.debug("Executing GET /ready against FastAPI inference service");
        try {
            FastApiReadinessResponse response = fastApiRestClient.get()
                    .uri("/ready")
                    .accept(MediaType.APPLICATION_JSON)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, (req, resp) -> {
                        throw new DownstreamContractViolationException(
                                "FastAPI readiness check failed with status: " + resp.getStatusCode());
                    })
                    .body(FastApiReadinessResponse.class);

            if (response == null || !"READY".equalsIgnoreCase(response.getStatus())) {
                throw new DownstreamContractViolationException("FastAPI returned non-ready status: " + response);
            }
            return response;
        } catch (Exception e) {
            handleClientException("checkReadiness", e);
            throw new FastApiUnavailableException("Failed to check FastAPI readiness", e);
        }
    }

    @Override
    public FastApiVersionResponse getVersion() {
        log.debug("Executing GET /version against FastAPI inference service");
        try {
            FastApiVersionResponse response = fastApiRestClient.get()
                    .uri("/version")
                    .accept(MediaType.APPLICATION_JSON)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, (req, resp) -> {
                        throw new DownstreamContractViolationException(
                                "FastAPI version check failed with status: " + resp.getStatusCode());
                    })
                    .body(FastApiVersionResponse.class);

            if (response == null) {
                throw new DownstreamContractViolationException("FastAPI returned null version response");
            }

            verifyModelIdentity(response.getModelId());
            return response;
        } catch (ModelIdentityMismatchException e) {
            throw e;
        } catch (Exception e) {
            handleClientException("getVersion", e);
            throw new FastApiUnavailableException("Failed to retrieve FastAPI version", e);
        }
    }

    @Override
    public FastApiPredictResponse predict(FastApiPredictRequest request) {
        log.debug("Executing POST /predict against FastAPI inference service with {} records",
                request != null && request.getRecords() != null ? request.getRecords().size() : 0);

        if (request == null || request.getRecords() == null || request.getRecords().isEmpty()) {
            throw new InferenceContractException("Prediction payload contains empty records list");
        }

        try {
            FastApiPredictResponse response = fastApiRestClient.post()
                    .uri("/predict")
                    .contentType(MediaType.APPLICATION_JSON)
                    .accept(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .onStatus(status -> status.value() == 400, (req, resp) -> {
                        throw new InferenceContractException(
                                "FastAPI rejected inference request: HTTP 400 Bad Request / Contract Violation");
                    })
                    .onStatus(HttpStatusCode::isError, (req, resp) -> {
                        throw new DownstreamContractViolationException(
                                "FastAPI inference execution failed with HTTP status: " + resp.getStatusCode());
                    })
                    .body(FastApiPredictResponse.class);

            if (response == null) {
                throw new DownstreamContractViolationException("FastAPI returned null prediction response");
            }

            if (!"SUCCESS".equalsIgnoreCase(response.getStatus())) {
                throw new DownstreamContractViolationException(
                        "FastAPI returned non-success prediction status: " + response.getStatus());
            }

            if (response.getForecasts() == null || response.getForecasts().isEmpty()) {
                throw new DownstreamContractViolationException("FastAPI returned empty forecasts list");
            }

            verifyModelIdentity(response.getModelVersion());
            return response;

        } catch (InferenceContractException | ModelIdentityMismatchException | DownstreamContractViolationException e) {
            throw e;
        } catch (Exception e) {
            handleClientException("predict", e);
            throw new FastApiUnavailableException("FastAPI inference call failed", e);
        }
    }

    private void verifyModelIdentity(String observedModelId) {
        String expectedModelId = properties.getFastApi().getExpectedModelId();
        if (properties.getFastApi().isFailOnModelIdMismatch() && expectedModelId != null && !expectedModelId.isBlank()) {
            if (observedModelId == null || !expectedModelId.equals(observedModelId)) {
                log.error("CRITICAL IMMUTABILITY VIOLATION: FastAPI reported model ID '{}', expected '{}'",
                        observedModelId, expectedModelId);
                throw new ModelIdentityMismatchException(
                        String.format("Downstream model identity mismatch! Expected '%s', but received '%s'. FAIL CLOSED.",
                                expectedModelId, observedModelId));
            }
        }
    }

    private void handleClientException(String operation, Exception e) {
        if (e instanceof ResourceAccessException rae) {
            Throwable cause = rae.getCause();
            if (cause instanceof HttpTimeoutException || cause instanceof HttpConnectTimeoutException) {
                log.error("FastAPI timeout during operation '{}': {}", operation, e.getMessage());
                throw new FastApiTimeoutException("Timeout connecting to FastAPI service during " + operation, e);
            }
            log.error("FastAPI connection failed during operation '{}': {}", operation, e.getMessage());
            throw new FastApiUnavailableException("FastAPI service is unavailable at configured base URL", e);
        } else if (e instanceof RestClientResponseException rcre) {
            log.error("FastAPI returned HTTP error during '{}': Status {}, Response: {}",
                    operation, rcre.getStatusCode(), rcre.getResponseBodyAsString());
            if (rcre.getStatusCode().value() == 400) {
                throw new InferenceContractException("FastAPI rejected contract: " + rcre.getResponseBodyAsString(), rcre);
            }
            throw new DownstreamContractViolationException("FastAPI HTTP " + rcre.getStatusCode() + ": " + rcre.getStatusText(), rcre);
        }
    }
}
