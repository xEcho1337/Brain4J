package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.GELU;

/**
 * Codec for {@link GELU}.
 */
public class GELUCodec implements JsonCodec<GELU> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "gelu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<GELU> targetClass() {
        return GELU.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(GELU gelu, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public GELU parse(JsonNode in) {
        return new GELU();
    }
}
