package com.atmosiq.exception;

public class FastApiUnavailableException extends AtmosIQException {
    public FastApiUnavailableException(String message) {
        super("DOWNSTREAM_SERVICE_UNAVAILABLE", message);
    }

    public FastApiUnavailableException(String message, Throwable cause) {
        super("DOWNSTREAM_SERVICE_UNAVAILABLE", message, cause);
    }
}
