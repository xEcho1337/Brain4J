package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.DropoutLayer;

/**
 * Codec for {@link DropoutLayer}.
 */
public class DropoutCodec implements JsonCodec<DropoutLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "dropout";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<DropoutLayer> targetClass() {
        return DropoutLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(DropoutLayer dropoutLayer, ObjectNode out) {
        out.put("rate", dropoutLayer.config().dropoutRate());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DropoutLayer parse(JsonNode in) {
        double rate = in.get("rate").asDouble();
        return new DropoutLayer(rate);
    }
}
