package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.LeakyReLU;

/**
 * Codec for {@link LeakyReLU}.
 */
public class LeakyReLUCodec implements JsonCodec<LeakyReLU> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "leaky_relu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<LeakyReLU> targetClass() {
        return LeakyReLU.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(LeakyReLU leakyReLU, ObjectNode out) {
        out.put("alpha", leakyReLU.alpha());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public LeakyReLU parse(JsonNode in) {
        double alpha = in.get("alpha").asDouble(0.01);
        return new LeakyReLU(alpha);
    }
}
