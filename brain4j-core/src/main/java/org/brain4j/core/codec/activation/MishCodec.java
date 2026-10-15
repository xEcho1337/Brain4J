package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Mish;

/**
 * Codec for {@link Mish}.
 */
public class MishCodec implements JsonCodec<Mish> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "mish";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<Mish> targetClass() {
        return Mish.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(Mish mish, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Mish parse(JsonNode in) {
        return new Mish();
    }
}
