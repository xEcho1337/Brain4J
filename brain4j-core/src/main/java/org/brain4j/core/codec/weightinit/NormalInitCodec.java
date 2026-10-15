package org.brain4j.core.codec.weightinit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.weightsinit.impl.NormalInit;

/**
 * Codec for {@link NormalInit}.
 */
public class NormalInitCodec implements JsonCodec<NormalInit> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "normal";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<NormalInit> targetClass() {
        return NormalInit.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(NormalInit normalInit, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NormalInit parse(JsonNode in) {
        return new NormalInit();
    }
}
