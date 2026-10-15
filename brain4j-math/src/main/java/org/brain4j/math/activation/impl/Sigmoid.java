package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

/**
 * Logistic sigmoid. Use it for binary outputs or gating units.
 */
public class Sigmoid implements Activation {

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
        return 1 / (1 + Math.exp(-input));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        double activated = activate(input);

        return activated * (1 - activated);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getKernelPrefix() {
        return "sigmoid";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getActivationId() {
        return 3;
    }
}
