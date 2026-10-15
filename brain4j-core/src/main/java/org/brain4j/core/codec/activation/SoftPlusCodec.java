package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.SoftPlus;

/**
 * Codec for {@link SoftPlus}.
 */
public class SoftPlusCodec implements JsonCodec<SoftPlus> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "softplus";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<SoftPlus> targetClass() {
        return SoftPlus.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(SoftPlus softplus, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SoftPlus parse(JsonNode in) {
        return new SoftPlus();
    }
}
