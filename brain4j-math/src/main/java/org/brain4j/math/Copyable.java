package org.brain4j.math;

/**
 * Creates an independent copy of an object.
 * Use it for value-like types that need duplicating before mutation.
 *
 * @param <T> the copyable type
 */
public interface Copyable<T> {

    /**
     * Returns an independent copy of this object.
     *
     * @return a new instance equal to this one
     */
    T copy();
}
