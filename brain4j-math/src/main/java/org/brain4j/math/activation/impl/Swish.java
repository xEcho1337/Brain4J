package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.NormalXavierInit;

/**
 * Swish activation. Use it for deep networks needing smooth gating.
 */
public class Swish implements Activation {

    /**
     * {@inheritDoc}
     */
    @Override
    public WeightInit defaultWeightInit() {
        return new NormalXavierInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double activate(double input) {
        return input * (1.0 / (1.0 + Math.exp(-input)));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        double sigmoid = 1.0 / (1.0 + Math.exp(-input));

        return sigmoid + input * sigmoid * (1 - sigmoid);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getKernelPrefix() {
        return "swish";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getActivationId() {
        return 5;
    }
}
