package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Tanh;

/**
 * Codec for {@link Tanh}.
 */
public class TanhCodec implements JsonCodec<Tanh> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "tanh";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<Tanh> targetClass() {
        return Tanh.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(Tanh tanh, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tanh parse(JsonNode in) {
        return new Tanh();
    }
}
