package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Swish;

/**
 * Codec for {@link Swish}.
 */
public class SwishCodec implements JsonCodec<Swish> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "swish";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<Swish> targetClass() {
        return Swish.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(Swish swish, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Swish parse(JsonNode in) {
        return new Swish();
    }
}
