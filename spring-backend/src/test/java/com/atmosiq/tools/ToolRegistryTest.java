package com.atmosiq.tools;

import com.atmosiq.config.AtmosIQProperties;
import com.atmosiq.exception.UnauthorizedToolException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ToolRegistryTest {

    private AtmosIQProperties properties;
    private ToolContract<String, String> dummyTool;
    private ToolRegistry registry;

    @BeforeEach
    void setUp() {
        properties = new AtmosIQProperties();
        properties.getOrchestration().setEnforceToolAllowlist(true);
        properties.getOrchestration().setAllowlistedTools(List.of("forecast_pm25", "dummy_allowed_tool"));

        dummyTool = new ToolContract<>() {
            @Override
            public String getName() {
                return "dummy_allowed_tool";
            }

            @Override
            public String getDescription() {
                return "Allowed test tool";
            }

            @Override
            public String execute(String request) {
                return "result:" + request;
            }
        };

        registry = new ToolRegistry(properties, List.of(dummyTool));
    }

    @Test
    @DisplayName("ToolRegistry accepts registered allowlisted tool")
    void testGetAllowlistedTool_Success() {
        ToolContract<String, String> tool = registry.getAllowlistedTool("dummy_allowed_tool");
        assertThat(tool).isNotNull();
        assertThat(tool.getName()).isEqualTo("dummy_allowed_tool");
        assertThat(tool.execute("hello")).isEqualTo("result:hello");
    }

    @Test
    @DisplayName("ToolRegistry denies unknown or non-allowlisted tool (Security Guard)")
    void testGetAllowlistedTool_Unauthorized_ThrowsException() {
        assertThatThrownBy(() -> registry.getAllowlistedTool("arbitrary_system_shell_tool"))
                .isInstanceOf(UnauthorizedToolException.class)
                .hasMessageContaining("not in the active AtmosIQ tool allowlist");
    }

    @Test
    @DisplayName("ToolRegistry lists active allowlisted tools")
    void testListAvailableToolNames() {
        List<String> tools = registry.listAvailableToolNames();
        assertThat(tools).contains("dummy_allowed_tool");
        assertThat(tools).doesNotContain("arbitrary_system_shell_tool");
    }
}
