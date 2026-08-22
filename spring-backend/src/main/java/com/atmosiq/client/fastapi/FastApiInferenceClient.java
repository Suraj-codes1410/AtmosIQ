package com.atmosiq.client.fastapi;

import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.model.fastapi.FastApiPredictRequest;
import com.atmosiq.model.fastapi.FastApiPredictResponse;
import com.atmosiq.model.fastapi.FastApiReadinessResponse;
import com.atmosiq.model.fastapi.FastApiVersionResponse;

/**
 * Strongly typed client boundary communicating with the certified downstream FastAPI service.
 */
public interface FastApiInferenceClient {

    FastApiHealthResponse checkHealth();

    FastApiReadinessResponse checkReadiness();

    FastApiVersionResponse getVersion();

    FastApiPredictResponse predict(FastApiPredictRequest request);
}
