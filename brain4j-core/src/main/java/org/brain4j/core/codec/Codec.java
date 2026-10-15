package org.brain4j.core.codec;

/**
 * Base codec contract.
 * Exposes a stable {@code type} discriminator and the target class.
 */
public interface Codec<T> {
    /**
     * Return the type discriminator.
     *
     * @return type name
     */
    String type();
    /**
     * Return the target class.
     *
     * @return target class
     */
    Class<T> targetClass();
}
