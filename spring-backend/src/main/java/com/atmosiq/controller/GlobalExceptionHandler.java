package com.atmosiq.controller;

import com.atmosiq.exception.DownstreamContractViolationException;
import com.atmosiq.exception.FastApiTimeoutException;
import com.atmosiq.exception.FastApiUnavailableException;
import com.atmosiq.exception.InferenceContractException;
import com.atmosiq.exception.InvalidForecastRequestException;
import com.atmosiq.exception.ModelIdentityMismatchException;
import com.atmosiq.exception.ToolExecutionException;
import com.atmosiq.exception.UnauthorizedToolException;
import com.atmosiq.model.response.ErrorResponseDto;
import com.atmosiq.provenance.ProvenanceTracker;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Global exception handler providing clean, structured error responses without leaking stack traces or internal paths.
 */
@Slf4j
@RestControllerAdvice
@RequiredArgsConstructor
public class GlobalExceptionHandler {

    private final ProvenanceTracker provenanceTracker;

    @ExceptionHandler(FastApiUnavailableException.class)
    public ResponseEntity<ErrorResponseDto> handleUnavailable(FastApiUnavailableException e) {
        log.error("FastApiUnavailableException: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.SERVICE_UNAVAILABLE, e.getErrorCode(), e.getMessage(), null);
    }

    @ExceptionHandler(FastApiTimeoutException.class)
    public ResponseEntity<ErrorResponseDto> handleTimeout(FastApiTimeoutException e) {
        log.error("FastApiTimeoutException: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.GATEWAY_TIMEOUT, e.getErrorCode(), e.getMessage(), null);
    }

    @ExceptionHandler(ModelIdentityMismatchException.class)
    public ResponseEntity<ErrorResponseDto> handleModelMismatch(ModelIdentityMismatchException e) {
        log.error("CRITICAL MODEL MISMATCH: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.INTERNAL_SERVER_ERROR, e.getErrorCode(),
                "Downstream model failed cryptographic or identity verification. Service stopped.", null);
    }

    @ExceptionHandler({InvalidForecastRequestException.class, InferenceContractException.class})
    public ResponseEntity<ErrorResponseDto> handleInvalidRequest(Exception e) {
        log.warn("Invalid forecast request: {}", e.getMessage());
        String code = e instanceof InvalidForecastRequestException ifre ? ifre.getErrorCode() : "INFERENCE_CONTRACT_VIOLATION";
        return buildErrorResponse(HttpStatus.BAD_REQUEST, code, e.getMessage(), null);
    }

    @ExceptionHandler(UnauthorizedToolException.class)
    public ResponseEntity<ErrorResponseDto> handleUnauthorizedTool(UnauthorizedToolException e) {
        log.error("Security violation: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.FORBIDDEN, e.getErrorCode(), e.getMessage(), null);
    }

    @ExceptionHandler(DownstreamContractViolationException.class)
    public ResponseEntity<ErrorResponseDto> handleContractViolation(DownstreamContractViolationException e) {
        log.error("Downstream contract violation: {}", e.getMessage());
        return buildErrorResponse(HttpStatus.BAD_GATEWAY, e.getErrorCode(), e.getMessage(), null);
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponseDto> handleValidationErrors(MethodArgumentNotValidException e) {
        List<String> details = new ArrayList<>();
        for (FieldError fieldError : e.getBindingResult().getFieldErrors()) {
            details.add(fieldError.getField() + ": " + fieldError.getDefaultMessage());
        }
        log.warn("Payload validation failed: {}", details);
        return buildErrorResponse(HttpStatus.BAD_REQUEST, "VALIDATION_FAILED", "Input payload failed validation checks.", details);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponseDto> handleGenericException(Exception e) {
        log.error("Unhandled orchestration exception: {}", e.getMessage(), e);
        return buildErrorResponse(HttpStatus.INTERNAL_SERVER_ERROR, "INTERNAL_ORCHESTRATION_ERROR",
                "An unexpected internal error occurred during orchestration.", null);
    }

    private ResponseEntity<ErrorResponseDto> buildErrorResponse(
            HttpStatus status,
            String errorCode,
            String message,
            List<String> details
    ) {
        ErrorResponseDto errorDto = ErrorResponseDto.builder()
                .error(errorCode)
                .message(message)
                .requestId(provenanceTracker.generateRequestId())
                .correlationId(provenanceTracker.getCurrentCorrelationId())
                .timestampUtc(Instant.now().toString())
                .validationDetails(details)
                .build();

        return ResponseEntity.status(status).body(errorDto);
    }
}
