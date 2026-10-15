package org.brain4j.math.operations;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Range;
import org.brain4j.math.convolution.ConvolveProvider;
import org.brain4j.math.convolution.impl.NormalConvolveProvider;
import org.brain4j.math.convolution.impl.SIMDConvolveProvider;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;

/**
 * Runs 2D convolution via im2col and parallel dot products. Use it for CPU
 * convolution with configurable stride.
 */
public class ParallelConvolve {

    private static final int CORES = Runtime.getRuntime().availableProcessors();
    private static final ForkJoinPool POOL = new ForkJoinPool(CORES);
    
    /**
     * Convolve with unit stride.
     *
     * @param a input tensor
     * @param b filter tensor
     * @return convolution result
     */
    public static Tensor convolve(Tensor a, Tensor b) {
        return convolve(a, b, 1);
    }

    /**
     * Convolve with the given stride.
     *
     * @param a input tensor
     * @param b filter tensor
     * @param stride step between filter positions, must be positive
     * @return convolution result
     */
    public static Tensor convolve(Tensor a, Tensor b, int stride) {
        if (stride <= 0) {
            throw new IllegalArgumentException("Stride must be > 0. Got: " + stride);
        }

        while (a.rank() < 4) {
            a = a.unsqueeze();
        }

        while (b.rank() < 4) {
            b = b.unsqueeze();
        }

        ConvolveProvider provider = DeviceUtils.isSimdAvailable()
            ? new SIMDConvolveProvider()
            : new NormalConvolveProvider();

        int[] aShape = a.shape();
        int[] bShape = b.shape();

        boolean aHasBatch = aShape.length == 4;
        int batch = aHasBatch ? aShape[0] : 1;
        int inChannels = aShape[aShape.length - 3];
        int inHeight = aShape[aShape.length - 2];
        int inWidth = aShape[aShape.length - 1];

        int numFilters = bShape[0];
        int filterHeight = bShape[2];
        int filterWidth = bShape[3];

        int outHeight = (inHeight - filterHeight) / stride + 1;
        int outWidth = (inWidth - filterWidth) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw new IllegalArgumentException("Invalid convolution output shape with stride " + stride);
        }

        int patchSize = inChannels * filterHeight * filterWidth;
        int totalPatches = outHeight * outWidth;

        Tensor out = Tensors.zeros(batch, numFilters, outHeight, outWidth);
        Tensor filterFlat = b.reshape(numFilters, patchSize);

        float[] filterData = filterFlat.data();
        float[] outData = out.data();

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            Tensor inputBatch = (aHasBatch ? a.slice(Range.point(bIdx)) : a).squeeze(0);
            Tensor patchMatrix = Tensors.im2col(inputBatch, filterHeight, filterWidth, stride);
            float[] patchData = patchMatrix.data();

            List<RecursiveAction> tasks = new ArrayList<>();

            for (int f = 0; f < numFilters; f++) {
                int filterOffset = f * patchSize;
                int outBase = (bIdx * numFilters + f) * totalPatches;

                int blockSize = 4096;
                int blocks = (totalPatches + blockSize - 1) / blockSize;

                ConvolveProvider.PatchData data = new ConvolveProvider.PatchData(
                    filterData, patchData, outData, totalPatches, patchSize, filterOffset, outBase
                );

                for (int block = 0; block < blocks; block++) {
                    int finalBlock = block;
                    tasks.add(new RecursiveAction() {
                        /**
                         * {@inheritDoc}
                         */
                        @Override
                        protected void compute() {
                            int start = finalBlock * blockSize;
                            int end = Math.min(start + blockSize, totalPatches);
                            provider.dotBlock(start, end, data);
                        }
                    });
                }

            }

            POOL.submit(() -> ForkJoinTask.invokeAll(tasks)).join();
        }

        return out;
    }
}
