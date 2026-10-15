package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.LSTMLayer;

/**
 * Codec for {@link LSTMLayer}.
 */
public class LSTMCodec implements JsonCodec<LSTMLayer> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "lstm";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<LSTMLayer> targetClass() {
        return LSTMLayer.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(LSTMLayer lstmLayer, ObjectNode out) {
        out.put("hidden_dimension", lstmLayer.config().hiddenDimension());
        out.put("return_sequences", lstmLayer.config().returnSequences());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public LSTMLayer parse(JsonNode in) {
        int hiddenDimension = in.get("hidden_dimension").asInt();
        boolean returnSequences = in.get("return_sequences").asBoolean();
        return new LSTMLayer(hiddenDimension, returnSequences);
    }
}
