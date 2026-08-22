package com.atmosiq.test;

import com.atmosiq.model.request.FeatureRecordDto;
import com.atmosiq.model.request.ForecastRequestDto;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Test data fixtures producing valid 14-day 35-feature observation windows.
 */
public final class TestFixtures {

    public static List<Map<String, Object>> createValid14DaySequence() {
        List<Map<String, Object>> records = new ArrayList<>();
        for (int i = 0; i < 14; i++) {
            Map<String, Object> row = new HashMap<>();
            row.put("date", String.format("2024-01-%02d", i + 1));
            for (String feature : FeatureRecordDto.CERTIFIED_35_FEATURES) {
                row.put(feature, 50.0 + i * 2.0);
            }
            records.add(row);
        }
        return records;
    }

    public static ForecastRequestDto createValidForecastRequest() {
        return ForecastRequestDto.builder()
                .records(createValid14DaySequence())
                .horizon("24h")
                .build();
    }
}
