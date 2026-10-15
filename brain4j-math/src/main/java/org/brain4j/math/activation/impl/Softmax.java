package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

/**
 * Temperature-scaled softmax. Use it for multi-class probability outputs.
 */
public record Softmax(double temperature) implements Activation {

    /**
     * Creates a softmax with unit temperature.
     */
    public Softmax() {
        this(1.0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double activate(double input) {
        return 1.0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor activate(Tensor input) {
        return input.softmax(temperature);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        return 0.0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getKernelPrefix() {
        return "";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor derivative(Tensor input, Tensor output, Tensor gradOut) {
        if (gradOut == null) {
            return null;
        }

        Tensor dot = gradOut.times(output).sum(-1, true);

        return output.times(gradOut.minus(dot));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getActivationId() {
        // Softmax runs on a dedicated kernel, so no scalar id applies.
        return -1;
    }
}
