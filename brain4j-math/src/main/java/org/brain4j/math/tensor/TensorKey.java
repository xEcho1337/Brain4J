package org.brain4j.math.tensor;

import java.util.Arrays;

/**
 * Identifies a cached buffer by usage and shape. Use it as a key for device
 * buffer reuse.
 *
 * @param usage buffer usage tag
 * @param shape shape identifying the buffer
 */
public record TensorKey(Usage usage, int... shape) {

    /**
     * Create a defensive copy of the shape.
     */
    public TensorKey {
        shape = shape == null ? new int[0] : shape.clone();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int[] shape() {
        return shape.clone();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (!(obj instanceof TensorKey other)) {
            return false;
        }

        return usage == other.usage && Arrays.equals(shape, other.shape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        int result = usage != null ? usage.hashCode() : 0;
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }
}
