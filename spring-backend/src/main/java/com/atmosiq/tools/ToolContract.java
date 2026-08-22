package com.atmosiq.tools;

/**
 * Common contract for all allowlisted tools available to Spring AI and orchestration layers.
 *
 * @param <REQ>  Input request type
 * @param <RESP> Output response type
 */
public interface ToolContract<REQ, RESP> {

    String getName();

    String getDescription();

    RESP execute(REQ request);
}
