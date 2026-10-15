package org.brain4j.math.broadcast;

import java.util.Arrays;

import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;

/**
 * Fast paths for common broadcast shapes. Use it to extend with a new element-wise op.
 */
public abstract class BaseBroadcast implements BroadcastOperation {

    protected final boolean vector = DeviceUtils.isSimdAvailable();

    /**
     * {@inheritDoc}
     */
    @Override
    public final Tensor defaultOp(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shapeA, shapeB)) {
            if (vector) {
                sameShapeSimd(aData, bData);
            } else {
                sameShape(aData, bData);
            }

            return A;
        }

        if (shapeA.length == 2 && shapeB.length == 1 && shapeA[1] == shapeB[0]) {
            if (vector) {
                rowWiseSimd(aData, bData, shapeA[0], shapeA[1]);
            } else {
                rowWise(aData, bData, shapeA[0], shapeA[1]);
            }

            return A;
        }

        if (shapeA.length == 3) {
            int d0 = shapeA[0];
            int d1 = shapeA[1];
            int d2 = shapeA[2];
            int total = d0 * d1 * d2;

            // [a, b, c] op [c] broadcasts the last axis across leading rows.
            if (shapeB.length == 1 && shapeA[2] == shapeB[0]) {
                if (vector) {
                    rowWiseSimd(aData, bData, total / d2, d2);
                } else {
                    rowWise(aData, bData, total / d2, d2);
                }

                return A;
            }

            // [a, b, c] op [b, c] broadcasts one matrix across the batch.
            if (shapeB.length == 2 && shapeA[1] == shapeB[0] && shapeA[2] == shapeB[1]) {
                if (vector) {
                    rowWiseSimd(aData, bData, d0, d1 * d2);
                } else {
                    rowWise(aData, bData, d0, d1 * d2);
                }

                return A;
            }
        }

        // [B, F, H, W] op [F] broadcasts one bias value per feature map.
        if (isBiasShape(shapeA, shapeB)) {
            if (vector) {
                biasSimd(A, B);
            } else {
                bias(A, B);
            }

            return A;
        }

        return fallbackOp(A, B);
    }

    /**
     * Applies the op element-wise on equal shapes.
     */
    protected abstract void sameShape(float[] a, float[] b);

    /**
     * Applies the op element-wise on equal shapes using SIMD.
     */
    protected abstract void sameShapeSimd(float[] a, float[] b);

    /**
     * Applies the op row by row.
     *
     * @param batch row count
     * @param dim row length
     */
    protected abstract void rowWise(float[] a, float[] b, int batch, int dim);

    /**
     * Applies the op row by row using SIMD.
     *
     * @param batch row count
     * @param dim row length
     */
    protected abstract void rowWiseSimd(float[] a, float[] b, int batch, int dim);

    /**
     * Applies the op with a convolutional bias tensor.
     */
    protected abstract void bias(Tensor output, Tensor biasTensor);

    /**
     * Applies the op with a convolutional bias tensor using SIMD.
     */
    protected abstract void biasSimd(Tensor output, Tensor biasTensor);

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor fallbackOp(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        int[] broadcastedShape = broadcastShape(shapeA, shapeB);

        if (!Arrays.equals(shapeA, broadcastedShape)) {
            throw new IllegalArgumentException("Broadcast result does not match shape of A! A = " +
                Arrays.toString(shapeA) + ", B = " + Arrays.toString(shapeB));
        }

        int total = A.elements();
        int[] stridesB = B.strides();
        int[] index = new int[shapeA.length];

        for (int i = 0; i < total; i++) {
            unravelIndex(i, shapeA, index);

            int bIndex = 0;

            for (int d = 0; d < shapeA.length; d++) {
                int dimB = shapeB.length - shapeA.length + d;

                if (dimB >= 0 && shapeB[dimB] != 1) {
                    bIndex += index[d] * stridesB[dimB];
                }
            }

            aData[i] = scalar(aData[i], bData[bIndex]);
        }

        return A;
    }

    /**
     * Applies the scalar op to one element pair.
     *
     * @return the combined value
     */
    protected abstract float scalar(float a, float b);
}
