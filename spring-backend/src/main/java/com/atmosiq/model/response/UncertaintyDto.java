package com.atmosiq.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Conformal prediction uncertainty bounds.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UncertaintyDto {

    private Double lowerBound;
    private Double upperBound;
    private Double conformalHalfWidth;
    private Double confidenceLevel; // e.g. 0.90
    private String methodology; // "Conformal Prediction (Split-Conformal Empirical Calibration)"
    private String physicalDisclaimer; // "PREDICTION INTERVAL != GUARANTEED PHYSICAL UNCERTAINTY"
}
