package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.MaxPoolLayer;

/**
 * Codec for {@link MaxPoolLayer}.
 */
public class MaxPoolCodec implements JsonCodec<MaxPoolLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "max_pool";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<MaxPoolLayer> targetClass() {
        return MaxPoolLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(MaxPoolLayer layer, ObjectNode out) {
        out.put("stride", layer.config().stride());
        out.put("window_height", layer.config().windowHeight());
        out.put("window_width", layer.config().windowWidth());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MaxPoolLayer parse(JsonNode in) {
        int stride = in.get("stride").asInt();
        int wh = in.get("window_height").asInt();
        int ww = in.get("window_width").asInt();

        return new MaxPoolLayer(stride, wh, ww);
    }
}
