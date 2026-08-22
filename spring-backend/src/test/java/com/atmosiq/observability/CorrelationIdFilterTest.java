package com.atmosiq.observability;

import com.atmosiq.provenance.ProvenanceTracker;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import static org.assertj.core.api.Assertions.assertThat;

class CorrelationIdFilterTest {

    private ProvenanceTracker provenanceTracker;
    private CorrelationIdFilter filter;

    @BeforeEach
    void setUp() {
        provenanceTracker = new ProvenanceTracker();
        filter = new CorrelationIdFilter(provenanceTracker);
    }

    @Test
    @DisplayName("Filter preserves incoming X-Correlation-ID header and returns it in response")
    void testFilter_PreservesIncomingHeader() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader(CorrelationIdFilter.CORRELATION_ID_HEADER, "custom_trace_999");
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockFilterChain chain = new MockFilterChain();

        filter.doFilter(request, response, chain);

        assertThat(response.getHeader(CorrelationIdFilter.CORRELATION_ID_HEADER)).isEqualTo("custom_trace_999");
        assertThat(response.getHeader(CorrelationIdFilter.REQUEST_ID_HEADER)).isNotNull();
    }

    @Test
    @DisplayName("Filter generates new X-Correlation-ID header when missing")
    void testFilter_GeneratesWhenMissing() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest();
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockFilterChain chain = new MockFilterChain();

        filter.doFilter(request, response, chain);

        String generatedCorrId = response.getHeader(CorrelationIdFilter.CORRELATION_ID_HEADER);
        assertThat(generatedCorrId).isNotNull();
        assertThat(generatedCorrId).startsWith("corr_");
    }
}
