package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Structured error response. Ensures internal paths/stack traces are never leaked.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ErrorResponseDto {

    private String error;
    private String message;
    private String requestId;
    private String correlationId;
    private String timestampUtc;
    private List<String> validationDetails;
}
