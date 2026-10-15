package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.DenseLayer;

/**
 * Codec for {@link DenseLayer}.
 */
public class DenseCodec implements JsonCodec<DenseLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "dense";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<DenseLayer> targetClass() {
        return DenseLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(DenseLayer denseLayer, ObjectNode out) {
        out.put("dimension", denseLayer.config().outDimension());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DenseLayer parse(JsonNode in) {
        int dim = in.get("dimension").asInt();
        return new DenseLayer(dim);
    }
}
