package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.RMSNormLayer;

/**
 * Codec for {@link RMSNormLayer}.
 */
public class RMSNormCodec implements JsonCodec<RMSNormLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "rms_norm";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<RMSNormLayer> targetClass() {
        return RMSNormLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(RMSNormLayer layer, ObjectNode out) {
        out.put("epsilon", layer.config().epsilon());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public RMSNormLayer parse(JsonNode in) {
        double eps = in.get("epsilon").asDouble();
        return new RMSNormLayer(eps);
    }
}
