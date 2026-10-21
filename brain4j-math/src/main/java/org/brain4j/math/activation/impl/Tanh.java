package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

/**
 * Hyperbolic tangent. Use it for bounded recurrent or zero-centered outputs.
 */
public class Tanh implements Activation {

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
        return Math.tanh(input);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        double activated = Math.tanh(input);

        return 1.0 - activated * activated;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String kernelPrefix() {
        return "tanh";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int activationId() {
        return 4;
    }
}
