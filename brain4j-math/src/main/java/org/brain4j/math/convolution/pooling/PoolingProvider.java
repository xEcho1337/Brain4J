package org.brain4j.math.convolution.pooling;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Defines forward and backward pooling over spatial dimensions. Use a concrete
 * subclass to downsample feature maps on CPU or GPU.
 */
public abstract class PoolingProvider {

    protected final int stride;
    protected final int windowHeight;
    protected final int windowWidth;

    /**
     * Create a pooling operator with the given geometry.
     *
     * @param stride step between windows, must be positive
     * @param windowHeight pooling window height, must be positive
     * @param windowWidth pooling window width, must be positive
     */
    public PoolingProvider(int stride, int windowHeight, int windowWidth) {
        this.stride = stride;
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }

    /**
     * Pool the input, dispatching to CPU or GPU.
     *
     * @param input tensor with at least two spatial dimensions
     * @return pooled tensor
     */
    public Tensor pool(Tensor input) {
        if (input instanceof GpuTensor gpuInput) {
            return poolGPU(gpuInput);
        } else {
            return poolCPU(input);
        }
    }

    /**
     * Pool the input on CPU.
     *
     * @param input tensor with at least two spatial dimensions
     * @return pooled tensor
     */
    public abstract Tensor poolCPU(Tensor input);

    /**
     * Pool the input on GPU.
     *
     * @param input tensor with at least two spatial dimensions
     * @return pooled tensor
     */
    public abstract Tensor poolGPU(Tensor input);

    /**
     * Propagate a gradient through the pooling window.
     *
     * @param gradient upstream gradient matching the pooled shape
     * @param input original forward input
     * @return gradient with the input shape
     */
    public Tensor backward(Tensor gradient, Tensor input) {
        if (gradient instanceof GpuTensor gpuGrad && input instanceof GpuTensor gpuInput) {
            return backwardGPU(gpuGrad, gpuInput);
        } else {
            return backwardCPU(gradient, input);
        }
    }

    /**
     * Propagate a gradient on CPU.
     *
     * @param gradient upstream gradient matching the pooled shape
     * @param input original forward input
     * @return gradient with the input shape
     */
    public abstract Tensor backwardCPU(Tensor gradient, Tensor input);

    /**
     * Propagate a gradient on GPU.
     *
     * @param gradient upstream gradient matching the pooled shape
     * @param input original forward input
     * @return gradient with the input shape
     */
    public abstract Tensor backwardGPU(GpuTensor gradient, GpuTensor input);

}
