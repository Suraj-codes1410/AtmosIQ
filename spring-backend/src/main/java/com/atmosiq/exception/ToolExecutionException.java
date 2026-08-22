package com.atmosiq.exception;

public class ToolExecutionException extends AtmosIQException {
    public ToolExecutionException(String message) {
        super("TOOL_EXECUTION_FAILURE", message);
    }

    public ToolExecutionException(String message, Throwable cause) {
        super("TOOL_EXECUTION_FAILURE", message, cause);
    }
}
