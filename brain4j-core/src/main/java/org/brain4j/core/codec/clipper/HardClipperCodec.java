package org.brain4j.core.codec.clipper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.clipper.impl.HardClipper;

/**
 * Codec for {@link HardClipper}.
 */
public class HardClipperCodec implements JsonCodec<HardClipper> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "clamp";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<HardClipper> targetClass() {
        return HardClipper.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(HardClipper hardClipper, ObjectNode out) {
        out.put("bound", hardClipper.bound());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public HardClipper parse(JsonNode in) {
        double bound = in.get("bound").asDouble(5.0);
        return new HardClipper(bound);
    }
}
