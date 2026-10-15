package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Sigmoid;

/**
 * Codec for {@link Sigmoid}.
 */
public class SigmoidCodec implements JsonCodec<Sigmoid> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "sigmoid";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<Sigmoid> targetClass() {
        return Sigmoid.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(Sigmoid sigmoid, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Sigmoid parse(JsonNode in) {
        return new Sigmoid();
    }
}
