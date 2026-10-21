package org.brain4j.math.activation.impl;

import org.silicon.api.function.ComputeFunction;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.NormalHeInit;

/**
 * Exponential linear unit. Use it to keep negative outputs with smooth saturation.
 */
public record ELU(double alpha) implements Activation {

    /**
     * Creates an ELU with default alpha.
     */
    public ELU() {
        this(1.0);
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
        return input > 0 ? input : (alpha * Math.exp(input) - 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : (alpha * Math.exp(input));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String kernelPrefix() {
        return "elu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int activationId() {
        return 2;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public KernelFactory createKernel(ComputeFunction kernel, GpuTensor input, GpuTensor output) {
        return KernelFactory
            .create(kernel)
            .intVal(activationId())
            .floatVal((float) alpha)
            .intVal(input.size())
            .buffer(input.dataBuffer())
            .buffer(output.dataBuffer());
    }
}
