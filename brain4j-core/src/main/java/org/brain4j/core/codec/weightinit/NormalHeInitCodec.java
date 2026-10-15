package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.weightsinit.impl.NormalHeInit;

/**
 * Codec for {@link NormalHeInit}.
 */
public class NormalHeInitCodec implements JsonCodec<NormalHeInit> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "normal_he";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<NormalHeInit> targetClass() {
        return NormalHeInit.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(NormalHeInit normalHeInit, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NormalHeInit parse(JsonNode in) {
        return new NormalHeInit();
    }
}
