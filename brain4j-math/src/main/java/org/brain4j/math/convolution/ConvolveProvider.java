package org.brain4j.math.convolution;

/**
 * Computes dot-product blocks over convolution patches. Use it to plug different
 * scalar or vectorized kernels into the parallel convolution loop.
 */
public interface ConvolveProvider {
    /**
     * Holds one filter and the patch block processed by a dot-product call.
     *
     * @param filter underlying filter weights
     * @param patch flattened input patches
     * @param out output buffer receiving one dot product per patch
     * @param totalPatches total number of patches in the block
     * @param patchSize elements per patch
     * @param filterOffset offset of the filter within {@code filter}
     * @param outOffset offset of the first result within {@code out}
     */
    record PatchData(
        float[] filter,
        float[] patch,
        float[] out,
        int totalPatches,
        int patchSize,
        int filterOffset,
        int outOffset
    ) {}

    /**
     * Compute dot products for patches in the half-open range.
     *
     * @param start first patch index, inclusive
     * @param end last patch index, exclusive
     * @param data filter, patches, and output buffers
     */
    void dotBlock(int start, int end, PatchData data);
}
