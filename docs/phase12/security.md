# AtmosIQ Phase 12 — Security & Governance Specification

## 1. Security Principles & Threat Model

The LLM is treated as an **untrusted entity** with respect to system infrastructure:

1. **Zero Filesystem Access**: The LLM has zero direct access to local filesystem paths or model checkpoints.
2. **Zero Shell Execution**: No shell or command execution tools are registered or accessible.
3. **Zero Arbitrary HTTP Access**: The backend only communicates with the configured FastAPI base URL; arbitrary HTTP dispatching is blocked.
4. **Tool Allowlisting**: `ToolRegistry` strictly checks incoming tool calls against `atmosiq.orchestration.allowlisted-tools`. Unknown tool requests are denied immediately with `403 FORBIDDEN` (`UnauthorizedToolException`).
5. **Secret Externalization**: API keys, credentials, and passwords are never hardcoded and never exposed in responses or logs.
6. **Information Disclosure Prevention**: `GlobalExceptionHandler` sanitizes all error responses, suppressing stack traces, internal file paths, or infrastructure details from clients.
