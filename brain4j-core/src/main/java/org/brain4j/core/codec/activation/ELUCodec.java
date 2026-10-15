package org.brain4j.core.codec.activation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.activation.impl.ELU;

/**
 * Codec for {@link ELU}.
 */
public class ELUCodec implements JsonCodec<ELU> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "elu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<ELU> targetClass() {
        return ELU.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(ELU elu, ObjectNode out) {
        out.put("alpha", elu.alpha());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ELU parse(JsonNode in) {
        double alpha = in.get("alpha").asDouble();
        return new ELU(alpha);
    }
}
