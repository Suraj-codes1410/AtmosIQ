package com.atmosiq.tools;

import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.UnauthorizedToolException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry and allowlist enforcer for all AtmosIQ tools.
 * Denies execution of any tool not explicitly allowlisted in configuration.
 */
@Slf4j
@Component
public class ToolRegistry {

    private final AtmosIQProperties properties;
    private final Map<String, ToolContract<?, ?>> registeredTools = new ConcurrentHashMap<>();

    public ToolRegistry(AtmosIQProperties properties, List<ToolContract<?, ?>> tools) {
        this.properties = properties;
        for (ToolContract<?, ?> tool : tools) {
            registeredTools.put(tool.getName(), tool);
            log.info("Registered tool in AtmosIQ registry: '{}' -> {}", tool.getName(), tool.getDescription());
        }
    }

    public boolean isToolAllowlisted(String toolName) {
        if (!properties.getOrchestration().isEnforceToolAllowlist()) {
            return registeredTools.containsKey(toolName);
        }
        return properties.getOrchestration().getAllowlistedTools().contains(toolName)
                && registeredTools.containsKey(toolName);
    }

    @SuppressWarnings("unchecked")
    public <REQ, RESP> ToolContract<REQ, RESP> getAllowlistedTool(String toolName) {
        if (!isToolAllowlisted(toolName)) {
            log.error("SECURITY VIOLATION: Unauthorized or unknown tool invocation attempt: '{}'", toolName);
            throw new UnauthorizedToolException(
                    String.format("Tool '%s' is not in the active AtmosIQ tool allowlist. Access denied.", toolName));
        }
        return (ToolContract<REQ, RESP>) registeredTools.get(toolName);
    }

    public List<String> listAvailableToolNames() {
        return properties.getOrchestration().getAllowlistedTools().stream()
                .filter(registeredTools::containsKey)
                .toList();
    }
}
