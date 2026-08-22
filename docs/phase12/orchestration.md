# AtmosIQ Phase 12 — Orchestration Service Specification

## 1. Responsibilities

The `OrchestrationService` coordinates application workflows between incoming client requests, Spring AI tools, and the downstream certified inference microservice.

### Core Functions:
1. **Request Validation**: Enforces input sequence validation ($W \ge 14$ rows, non-null, and 35 prediction-safe feature presence) prior to downstream dispatch.
2. **Correlation & Request ID Management**: Extracts or creates `X-Correlation-ID` and `X-Request-ID` to maintain full observability across distributed tiers.
3. **Tool Dispatch & Allowlisting**: Ensures that only tools explicitly registered in `ToolRegistry` and present in `atmosiq.orchestration.allowlisted-tools` are executed.
4. **Provenance Propagation**: Enriches output DTOs with model identity, parameter count (849), candidate ID, release status (`FINAL_PRODUCTION_CERTIFIED`), and cryptographic SHA-256 fingerprint.
5. **Structured Error Handling**: Converts internal or HTTP client failures into structured `ErrorResponseDto` payloads with appropriate HTTP status codes (400, 403, 500, 502, 503, 504).
6. **Observability**: Tracks telemetry counters via `OrchestrationMetrics` for total requests, successful forecasts, failures, and execution latency.
