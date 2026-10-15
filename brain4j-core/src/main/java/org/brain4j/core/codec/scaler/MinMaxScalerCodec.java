package org.brain4j.core.codec.scaler;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.math.scaler.impl.MinMaxScaler;

/**
 * Codec for {@link MinMaxScaler}.
 */
public class MinMaxScalerCodec implements JsonCodec<MinMaxScaler> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "min_max";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<MinMaxScaler> targetClass() {
        return MinMaxScaler.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(MinMaxScaler scaler, ObjectNode out) {
        out.put("range_min", scaler.rangeMin());
        out.put("range_max", scaler.rangeMax());
        out.put("data_min", scaler.dataMin());
        out.put("data_max", scaler.dataMax());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MinMaxScaler parse(JsonNode in) {
        float rangeMin = (float) in.get("range_min").asDouble();
        float rangeMax = (float) in.get("range_max").asDouble();
        float dataMin = (float) in.get("data_min").asDouble();
        float dataMax = (float) in.get("data_max").asDouble();
        return new MinMaxScaler(rangeMin, rangeMax, dataMin, dataMax);
    }
}
