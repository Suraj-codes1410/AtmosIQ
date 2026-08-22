package com.atmosiq.tools.health;

import com.atmosiq.client.fastapi.FastApiInferenceClient;
import com.atmosiq.model.fastapi.FastApiHealthResponse;
import com.atmosiq.tools.ToolContract;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.function.Function;

@Slf4j
@Component("checkServiceHealthTool")
@RequiredArgsConstructor
public class HealthTool implements ToolContract<Map<String, Object>, FastApiHealthResponse>,
        Function<Map<String, Object>, FastApiHealthResponse> {

    public static final String TOOL_NAME = "check_service_health";
    public static final String TOOL_DESCRIPTION = "Checks operational health status of the certified downstream AtmosIQ forecasting service.";

    private final FastApiInferenceClient inferenceClient;

    @Override
    public String getName() {
        return TOOL_NAME;
    }

    @Override
    public String getDescription() {
        return TOOL_DESCRIPTION;
    }

    @Override
    public FastApiHealthResponse apply(Map<String, Object> unused) {
        return execute(unused);
    }

    @Override
    public FastApiHealthResponse execute(Map<String, Object> unused) {
        log.debug("Executing HealthTool");
        return inferenceClient.checkHealth();
    }
}
