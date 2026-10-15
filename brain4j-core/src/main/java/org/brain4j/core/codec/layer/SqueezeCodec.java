package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.utility.SqueezeLayer;

/**
 * Codec for {@link SqueezeLayer}.
 */
public class SqueezeCodec implements JsonCodec<SqueezeLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "squeeze";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<SqueezeLayer> targetClass() {
        return SqueezeLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(SqueezeLayer layer, ObjectNode out) {
        out.put("dimension", layer.config().dimension());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SqueezeLayer parse(JsonNode in) {
        int dim = in.get("dimension").asInt();
        return new SqueezeLayer(dim);
    }
}
