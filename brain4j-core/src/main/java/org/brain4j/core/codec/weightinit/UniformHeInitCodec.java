package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.weightsinit.impl.UniformHeInit;

/**
 * Codec for {@link UniformHeInit}.
 */
public class UniformHeInitCodec implements JsonCodec<UniformHeInit> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "uniform_he";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<UniformHeInit> targetClass() {
        return UniformHeInit.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(UniformHeInit uniformHeInit, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public UniformHeInit parse(JsonNode in) {
        return new UniformHeInit();
    }
}
