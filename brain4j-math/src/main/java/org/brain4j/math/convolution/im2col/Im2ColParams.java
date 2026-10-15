package org.brain4j.math.convolution.im2col;

/**
 * Carries input and output buffers plus geometry for an im2col transform.
 * Use it to share one immutable patch-extraction description across tasks.
 *
 * @param inputData source image buffer
 * @param resultData destination buffer holding flattened patches
 * @param inputBaseOffset base offset within {@code inputData}
 * @param resultBaseOffset base offset within {@code resultData}
 * @param channelCount number of input channels
 * @param inputHeight input height
 * @param inputWidth input width
 * @param filterHeight filter height
 * @param filterWidth filter width
 * @param outputHeight number of patch rows
 * @param outputWidth number of patch columns
 */
public record Im2ColParams(
    float[] inputData,
    float[] resultData,
    int inputBaseOffset,
    int resultBaseOffset,
    int channelCount,
    int inputHeight,
    int inputWidth,
    int filterHeight,
    int filterWidth,
    int outputHeight,
    int outputWidth
) {}