package com.atmosiq.exception;

import lombok.Getter;

/**
 * Base abstract exception for all AtmosIQ domain and orchestration errors.
 */
@Getter
public abstract class AtmosIQException extends RuntimeException {

    private final String errorCode;

    public AtmosIQException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public AtmosIQException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }
}
