package org.brain4j.core.codec;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * JSON specialization of {@link Codec}.
 * Backs {@code BrainFormat} persistence of layers and helpers.
 */
public interface JsonCodec<T> extends Codec<T> {
    /**
     * Serialize a value into JSON.
     */
    void write(T value, ObjectNode out);
    /**
     * Deserialize a value from JSON.
     *
     * @return deserialized value
     */
    T parse(JsonNode in);
}
