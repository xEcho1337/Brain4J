package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.utility.SelectLayer;

/**
 * Codec for {@link SelectLayer}.
 */
public class SelectCodec implements JsonCodec<SelectLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "select";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<SelectLayer> targetClass() {
        return SelectLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(SelectLayer layer, ObjectNode out) {
        out.put("index", layer.config().index());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SelectLayer parse(JsonNode in) {
        int idx = in.get("index").asInt();
        return new SelectLayer(idx);
    }
}
