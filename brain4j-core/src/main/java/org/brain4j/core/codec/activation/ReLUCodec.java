package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.ReLU;

/**
 * Codec for {@link ReLU}.
 */
public class ReLUCodec implements JsonCodec<ReLU> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "relu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<ReLU> targetClass() {
        return ReLU.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(ReLU relu, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ReLU parse(JsonNode in) {
        return new ReLU();
    }
}
