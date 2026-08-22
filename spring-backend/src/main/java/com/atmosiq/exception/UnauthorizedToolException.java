package com.atmosiq.exception;

public class UnauthorizedToolException extends AtmosIQException {
    public UnauthorizedToolException(String message) {
        super("UNAUTHORIZED_TOOL_ACCESS", message);
    }
}
