package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.Softmax;

/**
 * Codec for {@link Softmax}.
 */
public class SoftmaxCodec implements JsonCodec<Softmax> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "softmax";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<Softmax> targetClass() {
        return Softmax.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(Softmax softmax, ObjectNode out) {
        out.put("temperature", softmax.temperature());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Softmax parse(JsonNode in) {
        double temperature = in.get("temperature").asDouble();
        return new Softmax(temperature);
    }
}
