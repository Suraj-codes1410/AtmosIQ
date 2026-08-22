package com.atmosiq.exception;

public class InferenceContractException extends AtmosIQException {
    public InferenceContractException(String message) {
        super("INFERENCE_CONTRACT_VIOLATION", message);
    }

    public InferenceContractException(String message, Throwable cause) {
        super("INFERENCE_CONTRACT_VIOLATION", message, cause);
    }
}
