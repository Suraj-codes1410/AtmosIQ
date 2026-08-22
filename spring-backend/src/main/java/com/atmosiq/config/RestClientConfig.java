package com.atmosiq.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;

/**
 * RestClient configuration with robust timeout and connection bounds.
 */
@Configuration
public class RestClientConfig {

    @Bean
    public RestClient fastApiRestClient(AtmosIQProperties properties, RestClient.Builder builder) {
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(properties.getFastApi().getConnectTimeoutMs()))
                .build();

        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(Duration.ofMillis(properties.getFastApi().getReadTimeoutMs()));

        return builder
                .baseUrl(properties.getFastApi().getBaseUrl())
                .requestFactory(requestFactory)
                .build();
    }
}
