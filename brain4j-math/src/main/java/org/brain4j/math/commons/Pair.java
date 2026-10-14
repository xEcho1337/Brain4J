package org.brain4j.math.commons;

/**
 * Holds two related values of possibly different types.
 * Use it for returning paired results without a dedicated type.
 *
 * @param <K> the first value type
 * @param <V> the second value type
 */
public record Pair<K, V>(K first, V second) {
}
