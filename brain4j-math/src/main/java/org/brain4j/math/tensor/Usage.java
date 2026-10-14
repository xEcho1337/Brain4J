package org.brain4j.math.tensor;

/**
 * Identifies which part of a tensor a cached buffer refers to.
 * Use it when pooling or reusing native memory.
 */
public enum Usage {
    STRIDES,
    DATA,
    OTHER
}
