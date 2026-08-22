package com.atmosiq.exception;

public class ModelIdentityMismatchException extends AtmosIQException {
    public ModelIdentityMismatchException(String message) {
        super("MODEL_IDENTITY_MISMATCH", message);
    }
}
