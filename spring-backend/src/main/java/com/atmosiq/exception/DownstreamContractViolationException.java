package com.atmosiq.exception;

public class DownstreamContractViolationException extends AtmosIQException {
    public DownstreamContractViolationException(String message) {
        super("DOWNSTREAM_CONTRACT_VIOLATION", message);
    }

    public DownstreamContractViolationException(String message, Throwable cause) {
        super("DOWNSTREAM_CONTRACT_VIOLATION", message, cause);
    }
}
