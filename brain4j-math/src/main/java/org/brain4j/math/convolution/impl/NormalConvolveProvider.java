package org.brain4j.math.convolution.impl;

import org.brain4j.math.convolution.ConvolveProvider;

/**
 * Computes convolution dot products with scalar loops. Use it as the portable
 * fallback when vector intrinsics are unavailable.
 */
public class NormalConvolveProvider implements ConvolveProvider {

    /**
     * {@inheritDoc}
     */
    @Override
    public void dotBlock(int start, int end, PatchData data) {
        int patchSize = data.patchSize();
        int filterOffset = data.filterOffset();
        int outOffset = data.outOffset();

        float[] filterData = data.filter();
        float[] patchData = data.patch();
        float[] outData = data.out();

        for (int p = start; p < end; p++) {
            float sum = 0f;
            int rowOffset = p * patchSize;

            for (int i = 0; i < patchSize; i++) {
                sum += filterData[filterOffset + i] * patchData[rowOffset + i];
            }

            outData[outOffset + p] = sum;
        }

    }
}
