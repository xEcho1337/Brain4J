package org.brain4j.math.commons;

import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * JSON persistence contract. Use it to save and restore model state.
 */
public interface JsonAdapter {
    /**
     * Writes state into the given JSON object.
     */
    void serialize(ObjectNode object);

    /**
     * Reads state from the given JSON object.
     */
    void deserialize(ObjectNode object);
}
