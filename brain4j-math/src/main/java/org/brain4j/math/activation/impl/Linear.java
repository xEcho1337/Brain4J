package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

/**
 * Identity activation. Use it for regression outputs or linear layers.
 */
public class Linear implements Activation {

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
        return input;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String kernelPrefix() {
        return "linear";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int activationId() {
        return 9;
    }
}
