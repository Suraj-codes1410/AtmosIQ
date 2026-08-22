# AtmosIQ Phase 12 — Spring AI Integration Specification

## 1. LLM / Tool Isolation Principle

In AtmosIQ, the Large Language Model (LLM) layer is strictly separated from the deterministic atmospheric forecasting layer.

- **Deterministic Layer**: Performs tensor matrix multiplications, feature scaling, runtime calibration offset addition ($-5.06\text{ }\mu\text{g/m}^3$), and split-conformal interval calculations ($\pm 95.66\text{ }\mu\text{g/m}^3$).
- **Spring AI Layer**: Orchestrates user prompts, converts unstructured requests to structured function calls, and synthesizes policy narratives using verified tool outputs.

The LLM is **NEVER** the forecasting model and is never allowed to fabricate predictions or adjust conformal bounds.

---

## 2. Tool Beans & Schema Registration

Tools are exposed to Spring AI via `SpringAiConfig` as `@Bean` functions:
1. `Function<ForecastToolRequest, ForecastToolResponse> forecastPm25`
2. `Function<Map<String, Object>, FastApiHealthResponse> checkServiceHealth`
3. `Function<Map<String, Object>, ModelMetadataDto> getModelMetadata`

Spring AI automatically derives the JSON Schema from the annotated `ForecastToolRequest` DTO, including explicit property descriptions and required constraints.
