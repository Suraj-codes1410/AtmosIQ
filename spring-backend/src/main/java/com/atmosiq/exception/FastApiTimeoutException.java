package com.atmosiq.exception;

public class FastApiTimeoutException extends AtmosIQException {
    public FastApiTimeoutException(String message) {
        super("DOWNSTREAM_SERVICE_TIMEOUT", message);
    }

    public FastApiTimeoutException(String message, Throwable cause) {
        super("DOWNSTREAM_SERVICE_TIMEOUT", message, cause);
    }
}
