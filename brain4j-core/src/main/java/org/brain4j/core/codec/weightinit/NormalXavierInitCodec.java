package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.weightsinit.impl.NormalXavierInit;

/**
 * Codec for {@link NormalXavierInit}.
 */
public class NormalXavierInitCodec implements JsonCodec<NormalXavierInit> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "normal_xavier";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<NormalXavierInit> targetClass() {
        return NormalXavierInit.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(NormalXavierInit normalXavierInit, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NormalXavierInit parse(JsonNode in) {
        return new NormalXavierInit();
    }
}
