package com.atmosiq.exception;

public class InvalidForecastRequestException extends AtmosIQException {
    public InvalidForecastRequestException(String message) {
        super("INVALID_FORECAST_REQUEST", message);
    }
}
