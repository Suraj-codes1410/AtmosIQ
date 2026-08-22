package com.atmosiq.client;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.client.fastapi.RestClientFastApiInferenceClient;
import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.DownstreamContractViolationException;
import com.atmosiq.exception.InferenceContractException;
import com.atmosiq.exception.ModelIdentityMismatchException;
import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.model.fastapi.FastApiPredictRequest;
import com.atmosiq.model.fastapi.FastApiPredictResponse;
import com.atmosiq.model.fastapi.FastApiReadinessResponse;
import com.atmosiq.model.fastapi.FastApiVersionResponse;
import com.atmosiq.test.TestFixtures;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

class FastApiInferenceClientTest {

    private MockRestServiceServer mockServer;
    private FastApiInferenceClient client;
    private AtmosIQProperties properties;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl("http://localhost:8000");
        mockServer = MockRestServiceServer.bindTo(builder).build();
        RestClient restClient = builder.build();

        properties = new AtmosIQProperties();
        properties.getFastApi().setBaseUrl("http://localhost:8000");
        properties.getFastApi().setExpectedModelId("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0");
        properties.getFastApi().setExpectedModelSha256("fdc99f7ca4410f3d577e52718e35c956c97f368cd91a8b7f505ee23824085bac");
        properties.getFastApi().setFailOnModelIdMismatch(true);

        client = new RestClientFastApiInferenceClient(restClient, properties);
    }

    @Test
    @DisplayName("Health endpoint returns HEALTHY status")
    void testCheckHealth_Success() {
        String json = """
                {
                    "status": "HEALTHY",
                    "service": "AtmosIQ_Production_Service",
                    "model_loaded": true,
                    "timestamp_utc": "2026-08-23T00:00:00Z"
                }
                """;

        mockServer.expect(requestTo("http://localhost:8000/health"))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        FastApiHealthResponse resp = client.checkHealth();

        assertThat(resp).isNotNull();
        assertThat(resp.getStatus()).isEqualTo("HEALTHY");
        assertThat(resp.getModelLoaded()).isTrue();
        mockServer.verify();
    }

    @Test
    @DisplayName("Readiness endpoint returns READY status")
    void testCheckReadiness_Success() {
        String json = """
                {
                    "status": "READY",
                    "model_version": "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0",
                    "feature_count": 35,
                    "scaler_ready": true,
                    "calibration_ready": true,
                    "timestamp_utc": "2026-08-23T00:00:00Z"
                }
                """;

        mockServer.expect(requestTo("http://localhost:8000/ready"))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        FastApiReadinessResponse resp = client.checkReadiness();

        assertThat(resp).isNotNull();
        assertThat(resp.getStatus()).isEqualTo("READY");
        assertThat(resp.getFeatureCount()).isEqualTo(35);
        mockServer.verify();
    }

    @Test
    @DisplayName("Version endpoint verifies expected model identity")
    void testGetVersion_Success() {
        String json = """
                {
                    "model_id": "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0",
                    "candidate_id": "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_CANDIDATE_v1.0.0",
                    "architecture": "TCN",
                    "parameters": 849,
                    "model_sha256": "fdc99f7ca4410f3d",
                    "release_status": "RELEASE_CERTIFIED"
                }
                """;

        mockServer.expect(requestTo("http://localhost:8000/version"))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        FastApiVersionResponse resp = client.getVersion();

        assertThat(resp).isNotNull();
        assertThat(resp.getModelId()).isEqualTo("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0");
        assertThat(resp.getParameters()).isEqualTo(849);
        mockServer.verify();
    }

    @Test
    @DisplayName("Version endpoint rejects unexpected model identity (FAIL CLOSED)")
    void testGetVersion_ModelMismatch_ThrowsException() {
        String json = """
                {
                    "model_id": "UNEXPECTED_ROGUE_MODEL_v2.0.0",
                    "candidate_id": "UNKNOWN",
                    "architecture": "MLP",
                    "parameters": 5000,
                    "model_sha256": "0000000000000000",
                    "release_status": "UNOFFICIAL"
                }
                """;

        mockServer.expect(requestTo("http://localhost:8000/version"))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        assertThatThrownBy(() -> client.getVersion())
                .isInstanceOf(ModelIdentityMismatchException.class)
                .hasMessageContaining("Downstream model identity mismatch");

        mockServer.verify();
    }

    @Test
    @DisplayName("Predict endpoint successfully parses forecasts and uncertainty")
    void testPredict_Success() {
        String json = """
                {
                    "status": "SUCCESS",
                    "model_version": "AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0",
                    "execution_latency_ms": 2.15,
                    "batch_size": 1,
                    "forecasts": [
                        {
                            "prediction_id": "abc12345def67890",
                            "timestamp_utc": "2024-01-14",
                            "forecast_pm25": 115.42,
                            "lower_90": 19.76,
                            "upper_90": 211.08,
                            "conformal_half_width": 95.66
                        }
                    ]
                }
                """;

        mockServer.expect(requestTo("http://localhost:8000/predict"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        FastApiPredictRequest request = FastApiPredictRequest.builder()
                .records(TestFixtures.createValid14DaySequence())
                .build();

        FastApiPredictResponse resp = client.predict(request);

        assertThat(resp).isNotNull();
        assertThat(resp.getStatus()).isEqualTo("SUCCESS");
        assertThat(resp.getModelVersion()).isEqualTo("AtmosIQ_DL_TCN_CAL07_25_PRODUCTION_v1.0.0");
        assertThat(resp.getForecasts()).hasSize(1);
        assertThat(resp.getForecasts().get(0).getForecastPm25()).isEqualTo(115.42);
        assertThat(resp.getForecasts().get(0).getConformalHalfWidth()).isEqualTo(95.66);
        mockServer.verify();
    }

    @Test
    @DisplayName("Predict endpoint throws InferenceContractException on HTTP 400")
    void testPredict_Http400_ThrowsInferenceContractException() {
        mockServer.expect(requestTo("http://localhost:8000/predict"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST));

        FastApiPredictRequest request = FastApiPredictRequest.builder()
                .records(TestFixtures.createValid14DaySequence())
                .build();

        assertThatThrownBy(() -> client.predict(request))
                .isInstanceOf(InferenceContractException.class);

        mockServer.verify();
    }

    @Test
    @DisplayName("Predict endpoint throws DownstreamContractViolationException on HTTP 500")
    void testPredict_Http500_ThrowsDownstreamContractViolationException() {
        mockServer.expect(requestTo("http://localhost:8000/predict"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withStatus(HttpStatus.INTERNAL_SERVER_ERROR));

        FastApiPredictRequest request = FastApiPredictRequest.builder()
                .records(TestFixtures.createValid14DaySequence())
                .build();

        assertThatThrownBy(() -> client.predict(request))
                .isInstanceOf(DownstreamContractViolationException.class);

        mockServer.verify();
    }
}
