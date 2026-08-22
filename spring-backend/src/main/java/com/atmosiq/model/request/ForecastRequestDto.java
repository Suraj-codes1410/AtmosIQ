package com.atmosiq.model.request;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Application-level request for PM2.5 forecasting.
 * Requires a sequential window of at least 14 daily observation records.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ForecastRequestDto {

    @NotEmpty(message = "Observation records sequence cannot be empty")
    @Size(min = 14, message = "Sequence length must contain at least 14 daily rows (W=14)")
    private List<Map<String, Object>> records;

    private String horizon; // e.g. "24h" (optional, defaults to 24h)
}
