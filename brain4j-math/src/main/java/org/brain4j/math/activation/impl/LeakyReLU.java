package org.brain4j.math.activation.impl;

import org.silicon.api.function.ComputeFunction;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.NormalHeInit;

/**
 * Leaky rectified linear unit. Use it to avoid dead neurons on negative inputs.
 */
public record LeakyReLU(double alpha) implements Activation {

    /**
     * Creates a leaky ReLU with default slope.
     */
    public LeakyReLU() {
        this(0.01);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public WeightInit defaultWeightInit() {
        return new NormalHeInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double activate(double input) {
        return Math.max(alpha * input, input);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : alpha;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String kernelPrefix() {
        return "leakyrelu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int activationId() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public KernelFactory createKernel(ComputeFunction kernel, GpuTensor input, GpuTensor output) {
        return KernelFactory.create(kernel)
            .intVal(activationId())
            .floatVal((float) alpha)
            .intVal(input.size())
            .buffer(input.dataBuffer())
            .buffer(output.dataBuffer());
    }
}
