package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Linear;

/**
 * Codec for {@link Linear}.
 */
public class LinearCodec implements JsonCodec<Linear> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "linear";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<Linear> targetClass() {
        return Linear.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(Linear linear, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Linear parse(JsonNode in) {
        return new Linear();
    }
}
