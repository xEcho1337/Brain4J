package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.utility.ActivationLayer;
import org.brain4j.math.activation.impl.Linear;

/**
 * Codec for {@link ActivationLayer}.
 */
public class ActivationCodec implements JsonCodec<ActivationLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "activation";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<ActivationLayer> targetClass() {
        return ActivationLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(ActivationLayer activationLayer, ObjectNode out) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ActivationLayer parse(JsonNode in) {
        return new ActivationLayer(new Linear());
    }
}
