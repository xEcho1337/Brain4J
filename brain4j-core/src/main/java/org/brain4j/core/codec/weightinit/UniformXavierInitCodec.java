package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

/**
 * Codec for {@link UniformXavierInit}.
 */
public class UniformXavierInitCodec implements JsonCodec<UniformXavierInit> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "uniform_xavier";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<UniformXavierInit> targetClass() {
        return UniformXavierInit.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(UniformXavierInit uniformXavierInit, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public UniformXavierInit parse(JsonNode in) {
        return new UniformXavierInit();
    }
}
