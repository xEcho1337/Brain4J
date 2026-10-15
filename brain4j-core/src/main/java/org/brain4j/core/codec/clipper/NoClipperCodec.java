package org.brain4j.core.codec.clipper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.clipper.impl.NoClipper;

/**
 * Codec for {@link NoClipper}.
 */
public class NoClipperCodec implements JsonCodec<NoClipper> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "none";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<NoClipper> targetClass() {
        return NoClipper.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(NoClipper noClipper, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NoClipper parse(JsonNode in) {
        return new NoClipper();
    }
}
