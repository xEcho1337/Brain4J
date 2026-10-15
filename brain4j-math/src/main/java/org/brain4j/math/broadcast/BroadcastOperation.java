package org.brain4j.math.broadcast;

import java.util.Arrays;

import org.brain4j.math.tensor.Tensor;

/**
 * Element-wise op with NumPy-style broadcasting. Use it to fuse tensors of compatible shapes.
 */
public interface BroadcastOperation {

    /**
     * Applies the fast-path op in place on A.
     *
     * @return the mutated tensor A
     */
    Tensor defaultOp(Tensor A, Tensor B);

    /**
     * Applies the generic strided op in place on A.
     *
     * @return the mutated tensor A
     */
    Tensor fallbackOp(Tensor A, Tensor B);

    /**
     * Converts a flat index into per-dimension indices.
     */
    default void unravelIndex(int flatIndex, int[] shape, int[] result) {
        for (int i = shape.length - 1; i >= 0; i--) {
            result[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
    }

    /**
     * Computes the broadcast shape of two shapes.
     *
     * @return the broadcasted dimensions
     */
    default int[] broadcastShape(int[] a, int[] b) {
        int len = Math.max(a.length, b.length);
        int[] result = new int[len];

        for (int i = 0; i < len; i++) {
            int dimA = i >= len - a.length ? a[i - (len - a.length)] : 1;
            int dimB = i >= len - b.length ? b[i - (len - b.length)] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new IllegalArgumentException("Incompatible shapes for broadcasting: " +
                    Arrays.toString(a) + " vs " + Arrays.toString(b));
            }

            result[i] = Math.max(dimA, dimB);
        }

        return result;
    }

    /**
     * Checks for a convolutional bias shape.
     *
     * @return true for [B, F, H, W] with a matching [F] bias
     */
    default boolean isBiasShape(int[] a, int[] b) {
        if (a.length != 4) {
            return false;
        }

        if (b.length == 1 && b[0] == a[1]) {
            return true;
        }

        return b.length == 4 && b[0] == 1 && b[2] == 1 && b[3] == 1 && b[1] == a[1];
    }
}
